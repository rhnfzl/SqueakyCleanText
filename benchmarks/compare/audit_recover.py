"""Audit of the copies ``scorer.recover_dropped`` credits to SCT (PROTOCOL.md section 17, recovery rule).

    python -m benchmarks.compare.audit_recover --cache benchmarks/.cache/merge-default \\
        --datasets gretel redact --out audit.json [--notes notes.md --notes-datasets gretel]

Per SCT entry x dataset: every recovered copy, then a seeded sample of ``SAMPLE`` checked with
``scorer.traces`` for a visible trace: the value with accents folded, case folded and separators
removed inside the output with the same folding (``compact``), or any 2+ char word of it, folded, as
a whole word of the output (``word``). A copy passes when neither is found. The first audit (rule
without the folding) found ``compact`` traces, so ``dropped_copies`` now applies ``traces`` itself and
a rerun checks that none is left on the real outputs.
``on_gold`` counts sampled copies that overlap a gold span (credited recall; the rest are extra
predictions). ``--out`` holds counts only, no text. ``--notes`` writes the sampled values in
context for ``--notes-datasets`` only (private notes; never for datasets whose text may not be
published).
"""

import argparse
import json
import random
from pathlib import Path

from benchmarks.compare import entries as registry
from benchmarks.compare import scorer, split
from benchmarks.score import overlaps
from sct.utils.atomic_file import write_text_atomic

SAMPLE = 50
SEED = 13
ENTRIES = ("sct-default", "sct-gliner")


def audit(records: list[dict], outputs: dict[str, dict], labels: dict, sample: int = SAMPLE,
          seed: int = SEED) -> tuple[dict[str, int], list[tuple[dict, dict, dict[str, bool]]]]:
    """(counts, sampled copies): copies = (record, span, check)."""
    copies = []
    for r in records:
        out = outputs.get(r["id"], {})
        if "error" in out or "masked_text" not in out:
            continue
        spans = scorer.map_spans(out["spans"], labels)[0]
        for extra in scorer.dropped_copies(r["text"], out["masked_text"], spans):
            copies.append((r, extra))
    chosen = random.Random(seed).sample(copies, min(sample, len(copies)))  # noqa: S311 (audit sample)
    rows = [(r, e, scorer.traces(r["text"][e["start"]:e["end"]], outputs[r["id"]]["masked_text"])) for r, e in chosen]
    counts = {"recovered": len(copies), "sampled": len(rows),
              "passed": sum(not (c["compact"] or c["word"]) for _, _, c in rows),
              "failed_compact": sum(c["compact"] for _, _, c in rows),
              "failed_word": sum(c["word"] for _, _, c in rows),
              "on_gold": sum(any(overlaps(e, g) for g in r["spans"]) for r, e, _ in rows)}
    return counts, rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--cache", type=Path, required=True, help="runner cache holding <entry>/<dataset>-test.jsonl")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--notes", type=Path)
    parser.add_argument("--notes-datasets", nargs="*", default=[])
    args = parser.parse_args(argv)
    result, notes = {}, []
    for dataset in args.datasets:
        records = split.load(dataset, "test", args.n, split.SEED)
        for entry in ENTRIES:
            raw = args.cache / entry / f"{dataset}-test.jsonl"
            if not raw.exists():
                continue
            outputs = {o["id"]: o for o in map(json.loads, raw.read_text(encoding="utf-8").splitlines())}
            counts, rows = audit(records, outputs, registry.ENTRIES[entry]["labels"])
            result.setdefault(entry, {})[dataset] = counts
            if args.notes and dataset in args.notes_datasets:
                notes.append(f"\n## {entry} {dataset}: {json.dumps(counts)}\n")
                for r, e, c in rows:
                    gold = sorted({g["label"] for g in r["spans"] if overlaps(e, g)}) or ["(no gold)"]
                    ctx = r["text"][max(0, e["start"] - 60):e["end"] + 60].replace("\n", " ")
                    value = r["text"][e["start"]:e["end"]]
                    notes.append(f"- `{r['id']}` {e['label']} gold={','.join(gold)} {c} value=`{value}` context=`{ctx}`")
    report = {"sample": SAMPLE, "seed": SEED, "entries": result}
    write_text_atomic(args.out, json.dumps(report, indent=1, sort_keys=True) + "\n")
    if args.notes:
        write_text_atomic(args.notes, "# recover_dropped audit notes (private)\n" + "\n".join(notes) + "\n")


if __name__ == "__main__":
    main()
