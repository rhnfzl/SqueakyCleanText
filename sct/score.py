"""Per-record score columns and a corpus data card.

Score A (always, free) flattens ``result.signals`` into Arrow-friendly
``sct_*`` columns. Score B (off by default, :class:`ScoreConfig`) adds a
pluggable quality score computed on the ORIGINAL text and a PII-normalised
MinHash signature computed on the anonymised text. All values are value-free:
counts, shares, type names, codes, scores and hashes, never entity text.
"""

import json
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol
from collections.abc import Iterable, Sequence

import regex

# Column -> Arrow type name; a list means a list of that type.
SCORE_A_TYPES = {
    "sct_language": "string", "sct_language_score": "float64", "sct_pii_count": "int64",
    "sct_pii_share": "float64", "sct_pii_types": ["string"], "sct_warnings": ["string"],
}
SCORE_A_COLUMNS = tuple(SCORE_A_TYPES)
SCORE_B_COLUMNS = ("sct_quality", "sct_minhash")

# Placeholders count as one token; reversible ``<PERSON_3>`` normalises to ``<PERSON>``.
_PLACEHOLDER = regex.compile(r"<([A-Z][A-Z0-9_]*?)(?:_\d+)?>")
_WORD_TOKEN = regex.compile(r"<[A-Z][A-Z0-9_]*>|\w+")
_CHAR_TOKEN = regex.compile(r"<[A-Z][A-Z0-9_]*>|\S")
# Scripts written without spaces between words get character shingles.
_NO_SPACE_SCRIPT = regex.compile(
    r"[\p{Han}\p{Hiragana}\p{Katakana}\p{Thai}\p{Lao}\p{Khmer}\p{Myanmar}]"
)
_LETTER = regex.compile(r"\p{L}")
SHARE_BINS = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
_BIN_LABELS = ("0",) + tuple(f"({lo}, {hi}]" for lo, hi in zip(SHARE_BINS, SHARE_BINS[1:]))


class QualityScorer(Protocol):
    """Scores a batch of ORIGINAL (not anonymised) texts, one float each."""

    def __call__(self, texts: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class ScoreConfig:
    """Score B switches; every one is off by default."""

    quality: bool = False
    quality_model: str | None = None
    quality_revision: str | None = None
    minhash: bool = False
    minhash_num_perm: int = 128
    minhash_seed: int = 1
    minhash_shingle: int = 5
    minhash_min_tokens: int = 5

    def __post_init__(self):
        if self.minhash_num_perm < 1:
            raise ValueError("minhash_num_perm must be >= 1")
        if self.minhash_shingle < 1:
            raise ValueError("minhash_shingle must be >= 1")
        if self.minhash_min_tokens < 0:
            raise ValueError("minhash_min_tokens must be >= 0")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def minhash_tokens(text: str) -> list[str]:
    """Casefolded tokens; char tokens when most letters are in a no-space script."""
    text = _PLACEHOLDER.sub(lambda m: f"<{m.group(1)}>", text)
    letters = len(_LETTER.findall(_PLACEHOLDER.sub(" ", text)))
    no_space = len(_NO_SPACE_SCRIPT.findall(text))
    pattern = _CHAR_TOKEN if letters and no_space * 2 > letters else _WORD_TOKEN
    return [t if t.startswith("<") else t.casefold() for t in pattern.findall(text)]


def minhash_signature(text: str, config: ScoreConfig) -> list[int] | None:
    """MinHash of ``config.minhash_shingle``-token shingles of anonymised text.

    ``None`` when fewer than ``minhash_min_tokens`` non-placeholder tokens.
    """
    try:
        from datasketch import MinHash
    except ImportError as exc:
        raise ImportError(
            "MinHash needs datasketch: pip install 'SqueakyCleanText[score]'"
        ) from exc
    tokens = minhash_tokens(text)
    if sum(not t.startswith("<") for t in tokens) < max(config.minhash_min_tokens, 1):
        return None
    size = config.minhash_shingle
    mh = MinHash(num_perm=config.minhash_num_perm, seed=config.minhash_seed)
    for i in range(max(len(tokens) - size + 1, 1)):
        mh.update("\x1f".join(tokens[i:i + size]).encode("utf-8"))
    return [int(v) for v in mh.hashvalues]


def minhash_metadata(config: ScoreConfig) -> dict[str, Any]:
    """Everything needed to compare signatures across runs."""
    return {
        "algo": "minhash",
        "lib": "datasketch",
        "lib_version": _package_version("datasketch"),
        "num_perm": config.minhash_num_perm,
        "seed": config.minhash_seed,
        "shingle": config.minhash_shingle,
        "min_tokens": config.minhash_min_tokens,
        "tokeniser": "char for Han/Hiragana/Katakana/Thai/Lao/Khmer/Myanmar text, word otherwise",
        "normaliser": "casefold; placeholders <TYPE_n> -> <TYPE>, one token each",
    }


def score_record(
    result,
    *,
    original: str | None = None,
    quality: QualityScorer | float | None = None,
    minhash: ScoreConfig | None = None,
) -> dict[str, Any]:
    """Flat ``sct_*`` columns for one ``ProcessResult``.

    Score A always. Both Score B columns are added when ``quality`` (a
    precomputed float, or a scorer run on ``original``) or ``minhash`` (a
    :class:`ScoreConfig`, run on ``result.lm_text``) is given.
    """
    signals = result.signals
    row: dict[str, Any] = {
        "sct_language": signals["language"],
        "sct_language_score": signals["language_confidence"],
        "sct_pii_count": signals["pii_count"],
        "sct_pii_share": signals["pii_share"],
        "sct_pii_types": sorted(signals["pii_types"]),
        "sct_warnings": list(signals["warnings"]),
    }
    if quality is None and minhash is None:
        return row
    if callable(quality):
        if original is None:
            raise ValueError("a quality scorer needs the original text")
        quality = quality([original])[0]
    row["sct_quality"] = None if quality is None else float(quality)
    row["sct_minhash"] = (
        None if minhash is None else minhash_signature(result.lm_text, minhash)
    )
    return row


def score_records(
    results: Sequence,
    originals: Sequence[str],
    config: ScoreConfig = ScoreConfig(),
    *,
    scorer: QualityScorer | None = None,
) -> list[dict[str, Any]]:
    """Score a batch; the quality scorer runs once over all originals.

    Passing ``scorer`` turns quality on, as ``ScoreConfig(quality=True)`` does.
    """
    if len(results) != len(originals):
        raise ValueError("results and originals must have the same length")
    qualities: list[float | None] = [None] * len(results)
    quality_on = config.quality or scorer is not None
    if quality_on:
        if scorer is None:
            if not config.quality_model:
                raise ValueError("quality=True needs quality_model or a scorer")
            scorer = ONNXQualityScorer(
                config.quality_model, revision=config.quality_revision,
            )
        qualities = list(scorer(list(originals))) if originals else []
    b_on = quality_on or config.minhash
    rows = []
    for result, original, q in zip(results, originals, qualities):
        row = score_record(result, minhash=config if config.minhash else None)
        if b_on:
            row["sct_quality"] = None if q is None else float(q)
            row.setdefault("sct_minhash", None)
        rows.append(row)
    return rows


class ONNXQualityScorer:
    """Quality head exported by ``scripts/export_quality_head.py``.

    ``model_dir`` holds ``model.onnx`` (token ids -> score in [0, 1]),
    ``tokenizer.json`` and ``sct_quality.json`` (sources and revisions).
    Only ONNX and JSON are loaded, never pickles. ``revision`` pins the head
    revision recorded at export time.
    """

    def __init__(self, model_dir, *, revision: str | None = None,
                 batch_size: int = 32, max_length: int = 512,
                 providers: list | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        path = Path(model_dir)
        manifest = path / "sct_quality.json"
        self.manifest = json.loads(manifest.read_text()) if manifest.is_file() else {}
        if revision is not None and self.manifest.get("head_revision") != revision:
            raise ValueError(
                f"quality model in {path} is revision "
                f"{self.manifest.get('head_revision')!r}, expected {revision!r}"
            )
        self.session = ort.InferenceSession(
            str(path / "model.onnx"),
            providers=providers or ["CPUExecutionProvider"],
        )
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        # shortcut: keeps the first max_length tokens of long documents; chunk and
        # average if long-document scores turn out to matter.
        self.tokenizer.enable_truncation(max_length)
        pad_id = self.tokenizer.token_to_id("<pad>")
        self.tokenizer.enable_padding(pad_id=pad_id if pad_id is not None else 0)
        self.batch_size = batch_size
        self._inputs = {i.name for i in self.session.get_inputs()}

    def __call__(self, texts: list[str]) -> list[float]:
        import numpy as np

        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            encoded = self.tokenizer.encode_batch(texts[start:start + self.batch_size])
            feeds = {
                "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.array(
                    [e.attention_mask for e in encoded], dtype=np.int64,
                ),
            }
            out = self.session.run(None, {k: v for k, v in feeds.items() if k in self._inputs})
            scores.extend(float(s) for s in np.asarray(out[0]).reshape(-1))
        return scores


def _share_bin(share: float) -> str:
    if share <= 0:
        return _BIN_LABELS[0]
    return next((label for high, label in zip(SHARE_BINS[1:], _BIN_LABELS[1:])
                 if share <= high), _BIN_LABELS[-1])


def build_data_card(
    rows: Iterable[dict[str, Any]],
    *,
    models: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Corpus summary from score rows plus the NER model provenance (K5)."""
    import sct

    rows = list(rows)
    languages: Counter = Counter()
    pii_types: Counter = Counter()
    warnings: Counter = Counter()
    shares: Counter = Counter()
    qualities = [r["sct_quality"] for r in rows if r.get("sct_quality") is not None]
    for r in rows:
        languages[r["sct_language"] or "unknown"] += 1
        pii_types.update(r["sct_pii_types"])
        warnings.update(r["sct_warnings"])
        shares[_share_bin(r["sct_pii_share"])] += 1
    unique_models = {m.get("key"): m for m in models}
    return {
        "sct_version": sct.__version__,
        "records": len(rows),
        "records_with_pii": sum(r["sct_pii_count"] > 0 for r in rows),
        "pii_items": sum(r["sct_pii_count"] for r in rows),
        "languages": dict(languages.most_common()),
        "pii_types": dict(sorted(pii_types.items())),
        "pii_share_histogram": {b: shares[b] for b in _BIN_LABELS if shares[b]},
        "warnings": dict(sorted(warnings.items())),
        "quality_mean": sum(qualities) / len(qualities) if qualities else None,
        "models": [unique_models[k] for k in sorted(unique_models, key=str)],
    }


def render_data_card(card: dict[str, Any]) -> str:
    """Markdown view of :func:`build_data_card`."""
    def table(title: str, counts: dict) -> list[str]:
        if not counts:
            return [f"## {title}", "", "None.", ""]
        return [f"## {title}", "", "| value | records |", "|---|---|",
                *(f"| {k} | {v} |" for k, v in counts.items()), ""]

    lines = [
        "# Data card", "",
        f"- SCT version: {card['sct_version']}",
        f"- Records: {card['records']}",
        f"- Records with PII: {card['records_with_pii']}",
        f"- PII items: {card['pii_items']}",
    ]
    if card.get("quality_mean") is not None:
        lines.append(f"- Mean quality: {card['quality_mean']:.4f}")
    lines.append("")
    lines += table("Languages", card["languages"])
    lines += table("PII types (records containing the type)", card["pii_types"])
    lines += table("PII share of text", card["pii_share_histogram"])
    lines += table("Warnings", card["warnings"])
    lines += ["## NER models", ""]
    lines += [
        f"- {m.get('key')}: {m.get('id')} @ {m.get('revision')} ({m.get('provider')})"
        for m in card["models"]
    ] or ["None (NER off)."]
    return "\n".join(lines) + "\n"
