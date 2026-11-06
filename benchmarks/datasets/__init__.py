"""Dataset loaders. Each module turns one open dataset into common records.

A record is ``{"id", "lang", "text", "spans": [{"start", "end", "label"}]}`` with
Python code-point offsets and the dataset's own label names (``benchmarks.labels``
maps them to SCT types). Every module sets ``NAME, HF_ID, REVISION, LICENCE,
PUBLISH, SUITE`` and ``load(n, seed)``: ``n`` docs per language (``None`` = all),
picked with a seeded reservoir so the sample only depends on the seed and the
pinned revision. Data is downloaded to ``benchmarks/.cache/`` (gitignored).
A loader may set ``record["part"]`` (``dev`` / ``test``) when the dataset ships its own split;
``compare.split`` keeps it instead of hashing.
"""

import hashlib
import importlib
import random
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

CACHE = Path(__file__).resolve().parents[1] / ".cache"

MODULES = (
    "openpii", "gretel", "nemotron", "redact", "iahlt", "bayanat", "thainer",
    "thai_pii", "naamapadam", "multiconer", "stockmark", "tab", "wili", "fineweb2", "gretel_en", "argilla_pii",
)
# Round 2 of the tool comparison (PROTOCOL.md section 1): fresh sets, none used while SCT was developed.
ROUND2 = (
    "newmean", "dialogpii", "kdpii", "nus_lt", "redmadrobot", "redactionbench", "tonic", "meddeid", "greek_deid",
    "neonredact", "pl_pii", "europii", "carmen",
)
MODULES = (*MODULES, *ROUND2)


def get(name: str):
    """Return the loader module for dataset *name*."""
    if name not in MODULES:
        raise ValueError(f"unknown dataset {name!r}; known: {', '.join(MODULES)}")
    return importlib.import_module(f"benchmarks.datasets.{name}")


def hf_file(repo: str, path: str, revision: str) -> Path:
    """Download one file of a HF dataset repo at a pinned revision (cached)."""
    return Path(hf_hub_download(repo, path, repo_type="dataset", revision=revision, cache_dir=CACHE / "hf"))


def hf_files(repo: str, patterns: list[str], revision: str) -> Path:
    """Download the files of a HF dataset repo matching *patterns* at a pinned revision; returns the snapshot dir."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, repo_type="dataset", revision=revision, allow_patterns=patterns,
                                  cache_dir=CACHE / "hf"))


def md5_checked(path: Path, md5: str) -> Path:
    """*path*, after checking its md5 against the pinned checksum (Zenodo records pin md5)."""
    digest = hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324  # integrity pin, not security
    if digest != md5:
        raise ValueError(f"{path.name}: md5 {digest} != pinned {md5}; delete it from the cache and retry")
    return path


def sha256_checked(path: Path, sha256: str | None) -> Path:
    """*path*, after checking its sha256 against the pin (SHORTLIST section 3); no pin, no check."""
    if sha256 is not None and (digest := hashlib.sha256(path.read_bytes()).hexdigest()) != sha256:
        raise ValueError(f"{path.name}: sha256 {digest} != pinned {sha256}; delete it from the cache and retry")
    return path


def url_file(url: str, name: str) -> Path:
    """Download *url* once to ``.cache/<name>``."""
    target = CACHE / name
    if not target.exists():
        if not url.startswith("https://"):
            raise ValueError(f"refusing non-https download: {url}")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url) as response, tmp.open("wb") as out:  # noqa: S310  # https only, checked above
            out.write(response.read())
        tmp.rename(target)
    return target


def sample_per_lang(records, n: int | None, seed: int) -> list[dict]:
    """Seeded reservoir sample of *n* records per ``lang``, in source order.

    Each language has its own RNG (seeded by ``seed`` and the language), so adding a
    language or changing another language's data never changes this one's sample.
    """
    kept: dict[str, list] = {}
    seen: dict[str, int] = {}
    rngs: dict[str, random.Random] = {}
    for index, record in enumerate(records):
        lang = record["lang"]
        if n is None:
            kept.setdefault(lang, []).append((index, record))
            continue
        count = seen.get(lang, 0)
        seen[lang] = count + 1
        bucket = kept.setdefault(lang, [])
        if count < n:
            bucket.append((index, record))
            continue
        rng = rngs.setdefault(lang, random.Random(f"{seed}:{lang}"))  # noqa: S311  # sampling, not crypto
        slot = rng.randrange(count + 1)
        if slot < n:
            bucket[slot] = (index, record)
    items = sorted((item for bucket in kept.values() for item in bucket), key=lambda item: item[0])
    return [record for _, record in items]


def bio_to_spans(tokens: list[str], tags: list[str], sep: str = " ") -> tuple[str, list[dict]]:
    """Join *tokens* with *sep* and turn BIO *tags* into char-offset spans.

    Tags are stripped (``" B-ORG"`` is ``B-ORG``). An ``I-`` tag that does not
    continue a span of the same type starts a new one (lenient IOB1).
    """
    text_parts, spans, offset, current = [], [], 0, None
    for i, (token, tag) in enumerate(zip(tokens, tags, strict=True)):
        if i:
            text_parts.append(sep)
            offset += len(sep)
        tag = str(tag).strip()
        start, end = offset, offset + len(token)
        text_parts.append(token)
        offset = end
        kind = tag[2:] if tag[:2] in ("B-", "I-") else None
        if kind and tag.startswith("I-") and current and current["label"] == kind:
            current["end"] = end
        elif kind:
            current = {"start": start, "end": end, "label": kind}
            spans.append(current)
        else:
            current = None
    return "".join(text_parts), spans


def valid_spans(text: str, spans: list[dict]) -> list[dict]:
    """Drop empty or out-of-bounds spans. A span may carry the source's surface ``form``: it must equal
    ``text[start:end]``, else the offset convention is wrong (UTF-16, inclusive end) and loading stops."""
    kept = [s for s in spans if 0 <= s["start"] < s["end"] <= len(text)]
    bad = sum(s.get("form") is not None and text[s["start"]:s["end"]] != s["form"] for s in kept)
    if bad:
        raise ValueError(f"{bad} of {len(kept)} gold spans do not match their surface form: check the offset convention")
    return [{k: v for k, v in s.items() if k != "form"} for s in kept]
