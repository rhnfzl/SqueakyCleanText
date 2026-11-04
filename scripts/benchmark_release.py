#!/usr/bin/env python3
"""Release benchmark: exact-span correctness, leaks and speed of the default cleaner.

Run once per release on the same machine and compare the JSON files:

    python -m scripts.benchmark_release

Batch throughput (``process_batch``) runs only with ``--batch N``:

    python -m scripts.benchmark_release --batch 256 --batch-size 8 --ner-batch-size 8

Correctness runs the default config with ``replacement_mode="reversible"`` (each
placeholder maps back to the exact value it replaced, which gives the predicted
span) and ``check_to_ascii_unicode=False`` (output stays in the source script, so
leaks are visible). Speed runs
the plain default config, single process, after one warmup pass over all documents.
"""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
import platform
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

from sct.evaluation import EntitySpan, evaluate_pii, load_evaluation_jsonl
from sct.utils.constants import LEGACY_TYPE_NAMES

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evaluation" / "release_benchmark.jsonl"
RESULTS = ROOT / "evaluation" / "results"
PLACEHOLDER_REGEX = re.compile(r"<([A-Z][A-Z_]*?)_\d+>")
NON_PII_LABELS = frozenset({"YEAR", "NUMBER"})
# Pre-0.8.0 reversible type names (legacy_type_names=True) -> dataset label.
LABEL_ALIASES = {old: new for new, old in LEGACY_TYPE_NAMES.items()}
WHITESPACE_RUN = re.compile(r"\s+")
ANY_PLACEHOLDER_REGEX = re.compile(r"<[A-Z][A-Z_]*(?:_\d+)?>")
SPEED_REPEATS = 3
CORRECTNESS_OVERRIDES = {"replacement_mode": "reversible", "check_to_ascii_unicode": False}
# A marker a regex step left for NER ("<PHONE>"); a NER span that cuts one takes it whole.
GENERIC_MARKER = re.compile(r"<([A-Z][A-Z_]*)>")


def _collapse_whitespace(view: str, offsets: list[int]) -> tuple[str, list[int]]:
    """*view* with each whitespace run as one space (SCT's normalize_whitespace), offsets kept per char."""
    chars, kept = [], []
    for ch, offset in zip(view, offsets):
        if ch.isspace():
            if chars and chars[-1] == " ":
                continue
            ch = " "
        chars.append(ch)
        kept.append(offset)
    return "".join(chars), kept + offsets[len(view):]


def _drop(view: str, offsets: list[int], pattern: re.Pattern, removed=lambda m: True,
          pieces: list[tuple[int, int]] | None = None) -> tuple[str, list[int]]:
    """*view* without the *pattern* matches SCT's ``pattern.sub`` deletes (*removed*), offsets kept per char.

    *pieces* are ranges searched on their own, as SCT removes brackets only outside JSON objects.
    """
    gone = set()
    for start, end in pieces or [(0, len(view))]:
        for match in pattern.finditer(view, start, end):
            if removed(match):
                gone.update(range(*match.span()))
    kept = [i for i in range(len(view)) if i not in gone]
    return "".join(view[i] for i in kept), [offsets[i] for i in kept] + offsets[len(view):]


def _sct_cleanup(view: str, offsets: list[int], cfg) -> tuple[str, list[int]]:
    """*view* without what SCT's pre-NER cleanup removes under *cfg*: isolated letters, bracket and
    brace content, isolated symbols and quotes, with the regexes and order of sct/utils/special.py."""
    from sct.utils import constants, special

    if cfg.check_remove_isolated_letters:
        view, offsets = _drop(view, offsets, constants.ISOLATED_LETTERS_REGEX)
    if not cfg.check_remove_isolated_special_symbols:
        return view, offsets
    max_chars, max_words = cfg.bracket_content_max_chars, cfg.bracket_content_max_words
    limited = max_chars > 0 or max_words > 0

    def short(match):  # special._remove_bracketed: with a limit, only short innermost content goes
        inner = match.group()[1:-1]
        return not ((0 < max_chars < len(inner)) or (0 < max_words < len(inner.split())))

    for on, whole, inner in ((cfg.check_remove_bracket_content, constants.BRACKET_CONTENT_REGEX,
                              constants.BRACKET_INNER_REGEX),
                             (cfg.check_remove_brace_content, constants.BRACE_CONTENT_REGEX,
                              constants.BRACE_INNER_REGEX)):
        if on:
            edges = [0, *(e for span in (special._json_object_spans(view) if cfg.keep_json_braces else [])
                          for e in span), len(view)]
            view, offsets = _drop(view, offsets, inner if limited else whole,
                                  short if limited else (lambda m: True), list(zip(edges[::2], edges[1::2])))
    view, offsets = _drop(view, offsets, constants.ISOLATED_SPECIAL_SYMBOLS_REGEX)
    return _drop(view, offsets, constants.ISOLATED_QUOTES_REGEX, lambda m: special._drop_if_isolated(m) == "")


def _find_value(value: str, views, low: int, high: int | None = None) -> tuple[int, int] | None:
    """Source offsets of the first and last char of the first *value* starting in source range
    [*low*, *high*), tried in each of *views* in turn; None if absent."""
    for text, offsets in views:
        at = text.find(value, bisect_left(offsets, low))
        if at >= 0 and (high is None or offsets[at] < high):
            return offsets[at], offsets[at + len(value) - 1]
    return None


def _nfkc_view(text: str) -> tuple[str, list[int]]:
    """NFKC form of *text*, plus the source offset of each of its characters.

    Each base character is normalised together with its combining marks, so every
    piece of the view maps back to one exact source range.
    shortcut: Hangul jamo sequences compose across clusters and fall back to -1 spans; fine
    unless a dataset ships decomposed Hangul.
    """
    pieces, offsets, start = [], [], 0
    for i in range(1, len(text) + 1):
        if i < len(text) and unicodedata.combining(text[i]):
            continue
        piece = unicodedata.normalize("NFKC", text[start:i])
        pieces.append(piece)
        offsets.extend([start] * len(piece))
        start = i
    offsets.append(len(text))
    return "".join(pieces), offsets


def predicted_spans(source: str, output: str, originals: dict[str, str],
                    view: tuple[str, list[int]] | None = None, cfg=None) -> list[EntitySpan]:
    """Map each ``<TAG_n>`` placeholder in output order to the source span it replaced.

    SCT may have folded the text (full-width to ASCII) before replacing it, so the
    search runs on the NFKC form of the source and the found range is mapped back
    to source offsets. *view* replaces that NFKC view with another ``(text, source offset
    of each char + [len(source)])`` pair, e.g. one that also decodes entities as SCT does.
    Whitespace runs compare as one space. NER values are searched first in the view without
    what SCT's pre-NER cleanup under *cfg* (default: ``TextCleanerConfig()``) removed ("Ann B Lee"
    holds "Ann Lee"), then in the view itself, where values regex steps took before that cleanup are.
    """
    if cfg is None:
        from sct.config import TextCleanerConfig

        cfg = TextCleanerConfig()
    base, base_offsets = view or _nfkc_view(source)
    views = (_collapse_whitespace(*_sct_cleanup(base, base_offsets, cfg)), _collapse_whitespace(base, base_offsets))
    view, offsets = views[0]
    shown = {m.group(0) for m in PLACEHOLDER_REGEX.finditer(output)}
    swallowed: dict[str, list[str]] = {}  # values whose placeholder a wider NER span took in, by tag
    for placeholder, original in originals.items():
        tag = PLACEHOLDER_REGEX.fullmatch(placeholder)
        if tag and placeholder not in shown:
            swallowed.setdefault(tag.group(1), []).append(original)
    output = WHITESPACE_RUN.sub(" ", unicodedata.normalize("NFKC", output))
    spans, cursor, previous_end = [], 0, 0  # cursor: source offset
    for match in PLACEHOLDER_REGEX.finditer(output):
        # Skip past the unchanged output text before this placeholder, so an
        # untouched earlier copy of the value ("Ann met <PERSON_0>") is not reused.
        context = output[previous_end:match.start()]
        previous_end = match.end()
        low = context_at = cursor
        # A cleanup step may have changed the context ("Ann x met" -> "Ann met"):
        # align on its longest suffix still found in the source.
        for cut in range(len(context)):
            found = view.find(context[cut:], bisect_left(offsets, cursor))
            if found >= 0:
                context_at, cursor = offsets[found], offsets[found + len(context) - cut]
                break
        label = LABEL_ALIASES.get(match.group(1), match.group(1))
        value = GENERIC_MARKER.sub(
            lambda m: swallowed[m.group(1)].pop(0) if swallowed.get(m.group(1)) else m.group(0),
            originals.get(match.group(0), ""))
        value = WHITESPACE_RUN.sub(" ", unicodedata.normalize("NFKC", value)).strip()
        # A context found too late moved the cursor past the value; the value still lies after the
        # previous one and before that context, never on text the output kept (an unmasked copy).
        found = (_find_value(value, views, cursor) or _find_value(value, views, low, context_at)) if value else None
        if found is None:
            # The replaced value is not in the source (an earlier step changed it): never matches.
            spans.append(EntitySpan(label, -1, -1))
            continue
        # A piece of the view can be several chars of one source char: end after that char.
        cursor = base_offsets[bisect_right(base_offsets, found[1])]
        spans.append(EntitySpan(label, found[0], cursor))
    return spans


def find_leaks(example, output: str) -> tuple[list[dict], list[dict]]:
    """Return PII values still fully present, and values with a piece left.

    A piece is a 3+ char word of the value still in the output, or text glued
    to a placeholder with 3+ letters or digits that is part of the value, so
    split values (``98765<NUMBER>``, ``<NUMBER>.1.10``) count.
    """
    leaks, partial = [], []
    visible = ANY_PLACEHOLDER_REGEX.sub("\0", output)
    fragments = [
        fragment.rstrip(".,;:!?)'\"")
        for fragment in re.findall(r"[^\s\0]+(?=\0)|(?<=\0)[^\s\0]+", visible)
    ]
    for span in example.expected:
        if span.label in NON_PII_LABELS:
            continue
        value = example.text[span.start:span.end]
        item = {"id": example.id, "label": span.label, "value": value}
        if value in output:
            leaks.append(item)
        elif any(
            len(token) >= 3 and token in visible for token in re.findall(r"\w+", value)
        ) or any(
            len(re.findall(r"\w", fragment)) >= 3 and fragment in value
            for fragment in fragments
        ):
            partial.append(item)
    return leaks, partial


def _leak_summary(items: list[dict], total: int) -> dict:
    return {
        "count": len(items),
        "total_pii": total,
        "by_label": dict(sorted(Counter(item["label"] for item in items).items())),
        "items": items,
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,  # noqa: S607
            text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run(dataset: Path) -> dict:
    import sct
    from sct.config import TextCleanerConfig
    from sct.sct import TextCleaner

    examples = load_evaluation_jsonl(dataset)

    cleaner = TextCleaner(cfg=TextCleanerConfig(**CORRECTNESS_OVERRIDES))
    outputs, predictions, leaks, partial = {}, {}, [], []
    for example in examples:
        result = cleaner.process(example.text)
        output = result.lm_text
        originals = {
            entry.placeholder: entry.original
            for entry in result.metadata["anon_map"].entries
        }
        outputs[example.id] = output
        predictions[example.id] = tuple(predicted_spans(example.text, output, originals, cfg=cleaner.cfg))
        doc_leaks, doc_partial = find_leaks(example, output)
        leaks += doc_leaks
        partial += doc_partial
    report = evaluate_pii(examples, predictions, model="default")
    pii_total = sum(
        span.label not in NON_PII_LABELS for e in examples for span in e.expected
    )
    del cleaner

    speed_cleaner = TextCleaner()
    for example in examples:
        speed_cleaner.process(example.text)
    durations = []
    for _ in range(SPEED_REPEATS):
        for example in examples:
            started_at = time.perf_counter()
            speed_cleaner.process(example.text)
            durations.append(time.perf_counter() - started_at)
    durations.sort()
    cfg = speed_cleaner.cfg

    return {
        "sct_version": sct.__version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset": dataset.name,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "documents": len(examples),
        "entities": sum(len(e.expected) for e in examples),
        "models": {
            "ner_backend": cfg.ner_backend,
            "ner_models": dict(cfg.ner_models or {}),
            "ner_model_revisions": dict(cfg.ner_model_revisions or {}),
        },
        "correctness": {
            "config": f"TextCleanerConfig(**{CORRECTNESS_OVERRIDES!r})",
            "report": report.to_dict(),
            "leaks": _leak_summary(leaks, pii_total),
            "partial_leaks": _leak_summary(partial, pii_total),
        },
        "speed": {
            "config": "TextCleanerConfig()",
            "warmup": "one pass over all documents",
            "repeats": SPEED_REPEATS,
            "timed_documents": len(durations),
            "seconds": round(sum(durations), 3),
            "docs_per_second": round(len(durations) / sum(durations), 3),
            "mean_ms": round(1000 * sum(durations) / len(durations), 3),
            "p95_ms": round(1000 * durations[max(0, -(-len(durations) * 95 // 100) - 1)], 3),
        },
        "outputs": outputs,
    }


def long_documents(texts: list[str], per_document: int = 8) -> list[str]:
    """Join every *per_document* consecutive texts into one document longer than a model window."""
    return [" ".join(texts[i:i + per_document]) for i in range(0, len(texts), per_document)]


def _peak_rss_mb() -> float | None:
    try:
        import resource  # noqa: PLC0415  # POSIX only
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)  # bytes vs KiB


def run_batch(
    dataset: Path,
    documents: int,
    batch_size: int | None,
    ner_batch_size: int,
    intra_op_threads: int | None,
    repeats: int,
    long: bool,
    **cfg_overrides,
) -> dict:
    """Time ``process_batch`` over *documents* docs (the dataset cycled) after one warmup call."""
    import sct
    from sct.config import TextCleanerConfig
    from sct.sct import TextCleaner

    texts = [example.text for example in load_evaluation_jsonl(dataset)]
    if long:
        texts = long_documents(texts)
    docs = [texts[i % len(texts)] for i in range(documents)]
    cleaner = TextCleaner(cfg=TextCleanerConfig(
        ner_batch_size=ner_batch_size, onnx_intra_op_threads=intra_op_threads, **cfg_overrides,
    ))
    cleaner.process_batch(docs, batch_size=batch_size)
    seconds = []
    for _ in range(repeats):
        started_at = time.perf_counter()
        results = cleaner.process_batch(docs, batch_size=batch_size)
        seconds.append(time.perf_counter() - started_at)
    output_hash = hashlib.sha256("\0".join(r.lm_text for r in results).encode()).hexdigest()
    return {
        "sct_version": sct.__version__,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dataset": dataset.name,
        "long_documents": long,
        "settings": {
            "documents": documents,
            "batch_size": batch_size,
            "ner_batch_size": ner_batch_size,
            "onnx_intra_op_threads": intra_op_threads,
            "repeats": repeats,
        },
        "seconds": [round(value, 3) for value in seconds],
        "docs_per_second": round(documents / min(seconds), 3),
        "peak_rss_mb": _peak_rss_mb(),
        "output_sha256": output_hash,
    }


def _row(name: str, metrics: dict) -> str:
    return (
        f"| {name} | {metrics['true_positives']} | {metrics['false_positives']} "
        f"| {metrics['false_negatives']} | {metrics['precision']:.2f} "
        f"| {metrics['recall']:.2f} | {metrics['f1']:.2f} |"
    )


def to_markdown(result: dict) -> str:
    report = result["correctness"]["report"]
    header = "| | TP | FP | FN | P | R | F1 |\n|---|---|---|---|---|---|---|"
    speed = result["speed"]
    leaks = result["correctness"]["leaks"]
    partial = result["correctness"]["partial_leaks"]
    lines = [
        f"# Release benchmark: SCT {result['sct_version']}",
        "",
        f"Commit `{result['git_commit']}`, {result['platform']}, Python {result['python']}.",
        f"Dataset `{result['dataset']}`: {result['documents']} documents, "
        f"{result['entities']} entities. Exact label + span match.",
        "",
        f"Speed (default config, single process): **{speed['docs_per_second']} docs/s**, "
        f"mean {speed['mean_ms']} ms, p95 {speed['p95_ms']} ms.",
        f"Leaks: **{leaks['count']} / {leaks['total_pii']}** PII values still in output "
        f"(partial: {partial['count']}).",
        "",
        "## By label",
        "",
        header,
        _row("**overall**", report["overall"]),
        *(_row(key, value) for key, value in report["by_label"].items()),
        "",
        "## By language",
        "",
        header,
        *(_row(key, value) for key, value in report["by_language"].items()),
        "",
        "## Leaks",
        "",
        *(f"- `{i['id']}` {i['label']}: {i['value']}" for i in leaks["items"]),
        *(f"- `{i['id']}` {i['label']} (partial): {i['value']}" for i in partial["items"]),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "Release benchmark").splitlines()[0])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    batch = parser.add_argument_group("batch throughput (only with --batch)")
    batch.add_argument("--batch", type=int, metavar="N", help="time process_batch over N documents")
    batch.add_argument("--batch-size", type=int, metavar="B", help="process_batch workers")
    batch.add_argument("--ner-batch-size", type=int, default=8, metavar="K")
    batch.add_argument("--intra-op-threads", type=int, metavar="T")
    batch.add_argument("--repeats", type=int, default=3, metavar="R")
    batch.add_argument("--long", action="store_true", help="join documents into long ones")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.batch:
        result = run_batch(
            args.dataset, args.batch, args.batch_size, args.ner_batch_size,
            args.intra_op_threads, args.repeats, args.long,
        )
        name = (
            f"benchmark-{result['sct_version']}-batch{'-long' if args.long else ''}"
            f"-N{args.batch}-B{args.batch_size}-K{args.ner_batch_size}-T{args.intra_op_threads}.json"
        )
        (args.output_dir / name).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        return
    result = run(args.dataset)
    stem = args.output_dir / f"benchmark-{result['sct_version']}"
    stem.with_name(stem.name + ".json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stem.with_name(stem.name + ".md").write_text(to_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
