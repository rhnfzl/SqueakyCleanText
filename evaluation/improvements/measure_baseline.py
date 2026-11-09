"""Measure pinned CPU NER on fixed artificial examples, without source values in results."""

import hashlib
import importlib
import importlib.metadata
import platform
import os
import subprocess
import sys
import json
from pathlib import Path
import resource
import statistics
import time

from sct.config import LANG_KEYS, TextCleanerConfig
from sct.utils.ner import GeneralNER

MODEL = "protectai/bert-base-NER-onnx"
REVISION = "84d28ab2bc28401b4353b26033eb043c8c8a0739"
SAMPLES = [
    ("John Smith works in Berlin.", {("PERSON", 0, 10), ("LOCATION", 20, 26)}),
    ("Angela Merkel visited Paris.", {("PERSON", 0, 13), ("LOCATION", 22, 27)}),
    ("The meeting starts tomorrow.", set()),
]


def instrument_sessions(pipelines):
    """Count each actual session call once, including shared pipeline aliases."""
    calls = [0]
    wrapped = {}

    class CountingSession:
        def __init__(self, session):
            self.session = session

        def run(self, *args, **kwargs):
            calls[0] += 1
            return self.session.run(*args, **kwargs)

    for pipe in pipelines:
        session = pipe._session
        if isinstance(session, CountingSession):
            continue
        if id(session) not in wrapped:
            wrapped[id(session)] = CountingSession(session)
        pipe._session = wrapped[id(session)]
    return calls


def source_identity(root):
    """Identify source bytes; archive revision declarations remain unverified."""
    root = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted((root / "sct").rglob("*.py")):
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big") + relative)
        digest.update(len(content).to_bytes(8, "big") + content)
    try:
        revision = subprocess.check_output(  # noqa: S603 - fixed Git commands, resolved source path or pytest-owned fixture path; no shell.
            ["/usr/bin/git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(  # noqa: S603 - fixed Git command and resolved source directory; no shell.
                ["/usr/bin/git", "-C", str(root), "status", "--porcelain", "--", "sct"], text=True
            )
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision, dirty = None, None
    return {
        "git_revision": revision,
        "dirty_source": dirty,
        "source_sha256": digest.hexdigest(),
        "declared_revision": os.environ.get("SCT_SOURCE_REVISION"),
    }


def main():
    cfg = TextCleanerConfig(
        ner_device="cpu", ner_models={k: MODEL for k in LANG_KEYS}, ner_model_revisions={k: REVISION for k in LANG_KEYS}
    )
    ner = GeneralNER(cfg)
    durations, true_positive, predicted, expected, exposed = [], 0, 0, 0, 0
    calls = instrument_sessions(ner._pipelines.values())
    for text, gold in SAMPLES:
        started = time.perf_counter()
        result = ner.ner_process_detailed(
            text, positional_tags=["PER", "ORG", "LOC"], ner_confidence_threshold=0.85, language="ENGLISH"
        )
        durations.append((time.perf_counter() - started) * 1000)
        found = {(f.entity.entity_type, f.entity.start, f.entity.end) for f in result.findings}
        true_positive += len(found & gold)
        predicted += len(found)
        expected += len(gold)
        exposed += bool(gold - found)
    output = {
        "source_identity": source_identity(Path(importlib.import_module("sct").__file__).resolve().parent.parent),
        "command": sys.argv,
        "measurement_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "HF_HUB_OFFLINE", "SCT_SOURCE_REVISION")
        },
        "model": MODEL,
        "revision": REVISION,
        "device": "cpu",
        "sample_count": len(SAMPLES),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "providers": {
                name: {"version": importlib.metadata.version(name), "module_path": importlib.import_module(name).__file__}
                for name in ("onnxruntime", "tokenizers", "numpy")
            },
        },
        "sample_sha256": hashlib.sha256(json.dumps([(text, sorted(gold)) for text, gold in SAMPLES]).encode()).hexdigest(),
        "config_sha256": hashlib.sha256(
            json.dumps(cfg._plain_state(), sort_keys=True, default=lambda value: sorted(value)).encode()
        ).hexdigest(),
        "true_positive": true_positive,
        "predicted": predicted,
        "expected": expected,
        "exact_span_precision": true_positive / predicted if predicted else None,
        "exact_span_recall": true_positive / expected,
        "document_exposure": exposed / len(SAMPLES),
        "median_ms": statistics.median(durations),
        "p95_ms": max(durations),
        "p95_method": "nearest rank; n=3",
        "model_calls": calls[0],
        "peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limits": "Three artificial English examples; not a representative quality or routing promotion gate.",
    }
    assert 0 <= true_positive <= expected and len(durations) == 3
    Path("baseline-measured.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
