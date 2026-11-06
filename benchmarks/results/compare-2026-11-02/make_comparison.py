"""Build COMPARISON.md from the bundle: tables from benchmarks.compare.report, prose around them.

    python benchmarks/results/compare-2026-11-02/make_comparison.py benchmarks/results/compare-2026-11-02 \
        benchmarks/results/compare-2026-11-02/comparison-tail.md > benchmarks/results/compare-2026-11-02/COMPARISON.md

Run from the repo root (``PYTHONPATH=.``). Numbers in the prose are read from the bundle here, never typed;
``comparison-tail.md`` is the static end of the report (feature matrix, limits, notes).
"""

import json
import sys
from pathlib import Path

from benchmarks.compare import entries as registry
from benchmarks.compare import report, scorer

bundle = Path(sys.argv[1])
S = {t: json.loads((bundle / t / "summary.json").read_text()) for t in ("default", "tuned")}
W = {t: json.loads((bundle / "norecover" / t / "summary.json").read_text()) for t in ("default", "tuned")}
ENTRIES = list(registry.ENTRIES)
GROUPS = list(scorer.GROUPS)
AGG = S["default"]["aggregate"]
out: list[str] = []


def emit(*lines):
    out.extend(lines)


def table(lines):
    emit(*([] if out and out[-1] == "" else [""]), *lines, "")


def res(track, entry, dataset, without=False):
    root = bundle / ("norecover/" + track if without else track)
    return json.loads((root / entry / f"{dataset}-test.json").read_text())


def mr(track, entry, dataset, without=False, metric="masked_recall"):
    return report.claimed_macro(res(track, entry, dataset, without)["views"]["full"]["compare"], entry, metric)


def left_clean() -> list[str]:
    rows = ["| entry | privacy-synthetic | privacy-real | NER | claimed languages | round 1 status |",
            "|---|---|---|---|---|---|"]
    for e in ENTRIES:
        cells, ranked = [], False
        for g in GROUPS:
            sets = [d for d in scorer.GROUPS[g] if registry.status(e, d) == "clean"]
            cell = S["default"]["own"].get(e, {}).get("full", {}).get(g, {}).get("clean") or {}
            used = (cell.get("claimed") or {}).get("datasets", [])
            ranked |= bool(used)
            outside = [d for d in sets if d not in used]
            text = ", ".join(used) or "none"
            if outside:
                text += f" (+ outside its languages only: {', '.join(outside)})"
            cells.append(text if sets else "none")
        unknown = any(registry.status(e, d) == "unknown" for g in GROUPS for d in scorer.GROUPS[g])
        status = ("has clean cells" if ranked else "**no clean cell: round 1 cannot rank it**"
                  + (" (U on every set)" if unknown and not e.startswith("sct") else ""))
        rows.append(f"| {e} | " + " | ".join(cells) + f" | {report.claimed(e)} | {status} |")
    return rows


def own(track, view, group):
    return report.own_table(S[track], bundle, view, group, W[track])


def same(track, view, group):
    return report.same_set_table(S[track], view, group, "others")


def recovered_on(track, entry, dataset):
    return res(track, entry, dataset).get("recovered_dropped_copies")


# ---------------------------------------------------------------- numbers used in the prose
gl_en_copies = recovered_on("tuned", "sct-gliner", "gretel_en")
gl_en_copies_d = recovered_on("default", "sct-gliner", "gretel_en")
audits = {t: json.loads((bundle / "audit" / f"{t}.json").read_text())["entries"] for t in ("default", "tuned")}
firsts = {t: json.loads((bundle / "audit" / f"{t}-first-pass.json").read_text())["entries"] for t in ("default", "tuned")}
sampled = sum(c["sampled"] for t in audits for e in audits[t].values() for c in e.values())
passed = sum(c["passed"] for t in audits for e in audits[t].values() for c in e.values())
first_traces = sum(c["sampled"] - c["passed"] for t in firsts for e in firsts[t].values() for c in e.values())
first_sampled = sum(c["sampled"] for t in firsts for e in firsts[t].values() for c in e.values())
sets_with_traces = sorted({(t, e, d) for t in firsts for e, v in firsts[t].items() for d, c in v.items()
                           if c["sampled"] > c["passed"]})
max_traces = max(c["sampled"] - c["passed"] for t in firsts for e in firsts[t].values() for c in e.values())
on_gold = sum(c["on_gold"] for t in audits for e in audits[t].values() for c in e.values())

# ---------------------------------------------------------------- document
emit("# SCT vs other PII tools: comparison, 2026-11-02 (rules 2026-11-03, generator families 2026-11-03)", "",
     "SqueakyCleanText 0.8.1 (`sct-default`, and its optional GLiNER path `sct-gliner`) against six",
     "other offline PII tools, under the frozen protocol in `benchmarks/compare/PROTOCOL.md`",
     "(Codex's 20 fairness rules; section 20 maps each rule). Numbers come from this bundle",
     "(`default/`, `tuned/`, `norecover/`, `audit/`, `speed-*/`); every table can be rebuilt with",
     "`python -m benchmarks.compare.report benchmarks/results/compare-2026-11-02` (see `TABLES.md`), and",
     "this file with `make_comparison.py` in the bundle (prose numbers are read from the bundle).", "",
     "- Entries (exact pins in `PROTOCOL.md` section 2): sct-default, sct-gliner, presidio 2.2.366",
     "  (en_core_web_lg), gliner-nvidia (nvidia/gliner-PII), gliner2-fastino",
     "  (fastino/gliner2-privacy-filter-PII-multi), opf (openai/privacy-filter), bardsai-v3",
     "  (bardsai/eu-pii-anonimization-multilang-v3, INT8 ONNX), datafog 4.9.0 (regex).",
     "- Data: test part (hash split by template/record, seed 13), 300 docs per language",
     "  (TAB 106, REDACT 825), 14 datasets, 16,661 docs per entry. Privacy-synthetic: openpii, gretel",
     "  (finance), gretel_en, nemotron, argilla_pii, thai_pii, bayanat. Privacy-real: TAB, REDACT.",
     "  NER (separate, not privacy): iahlt, thainer, naamapadam, multiconer, stockmark.",
     "- Tracks: **default** = vendor quick start (the primary track); **tuned** = one configuration per",
     "  entry picked from at most 12 candidates on the dev parts of all nine privacy sets, frozen in",
     "  `benchmarks/compare/configs/tuned.json` before the test run. Because the tuned choice saw the dev",
     "  part of every privacy set, tuned results are shown second. Gold-language track. All quality runs",
     "  on CPU. One mask operation for every entry.", "")

emit("## What changed on 2026-11-03", "",
     "Two passes, both rescoring the same raw outputs (no new model runs).", "",
     "1. **Rules pass.** Contamination (decision 21): if a tool, or any model inside it, was trained on a",
     "   dataset (T), is evaluated on it by its vendor (E), or SCT was developed against it (D), that cell",
     "   is excluded for that tool; unknown training data (U) is shown apart, never as clean. Claimed",
     "   languages: each tool is scored only on the languages its model card claims.",
     "2. **Families and recovery pass** (an independent check of pass 1):",
     "   - The rule covers a dataset **and its generator family**, for every entry, SCT included.",
     "     Families (`entries.FAMILIES`): Gretel (gretel finance, gretel_en, nemotron: Gretel Navigator",
     "     and its successor NeMo Data Designer) and ai4privacy (openpii, argilla_pii). So gretel_en and",
     "     argilla_pii are now D-family for SCT, gretel finance is T-family for bardsai-v3 (and",
     "     T-family, E-family for gliner-nvidia), nemotron E-family for gliner-nvidia.",
     "   - Pass 1 said gretel_en and argilla_pii were \"held out for SCT, dev included\". That was",
     "     wrong: the tuned track picked SCT's configuration (like every entry's) on the dev parts of",
     "     all privacy sets, these two included. The default track is now primary.",
     "   - Per-dataset tables use only each entry's claimed languages (`°` = no such doc; the rest",
     "     is a separate \"outside claimed languages\" table).",
     "   - The recovery rule for SCT's dropped placeholders was described as \"3+ char words\"; the code",
     "     used 4 (`LEAK_MIN_CHARS`). It is now stricter (any visible 2+ char word, case- and",
     "     accent-insensitive, or the value's separator-free form, blocks it) and was audited; every SCT",
     "     number it changes is shown with and without it.",
     f"   - Pass 1 said \"0 recovered copies\" on gretel_en and argilla_pii: sct-gliner had {gl_en_copies}",
     "     on gretel_en (tuned).",
     "   - sct-gliner's claim (en) is now taken from the model card's language tag; sct-default claims",
     "     every language and is held to every language. Licence notes corrected (end of report).", "")

emit("## Bottom line", "",
     "- **SCT has no clean cell in round 1.** sct-default is D or D-family on all 14 datasets, sct-gliner",
     "  U plus D or D-family on all 14. **Round 1 cannot rank SCT** (either entry), on any group. Every SCT",
     "  number below is information only. Round 2 needs fresh datasets that no tool and no SCT",
     "  development has seen, and that belong to no generator family any of them has seen.",
     "- **opf has no clean cell either** (U everywhere: training data undisclosed).",
     "- The other five tools are compared only with each other, on their own clean cells, in their",
     "  claimed languages. What each tool has left (default track, full view):", "")
table(left_clean())
emit("There is no SCT pairwise or same-set table in this report: no dataset is clean for SCT.", "")

emit("How to read the tables:", "",
     "- Scores are the group macro: mean over datasets of each dataset's language macro, failed docs",
     "  counted unmasked, DATE out. Synthetic and real data are never pooled.",
     "- mR = type-agnostic masked recall; mP = masked precision; strict F1 = exact span and type;",
     "  doc leak = share of docs with a gold mention that keeps any letter or digit unmasked;",
     "  over-mask = masked characters outside every gold span / all non-gold characters.",
     "- Own tables: the entry on its own clean datasets, only docs in its claimed languages, with a 95%",
     "  bootstrap CI (1,000 resamples of templates/docs within each dataset, seed 13). Beside it, for",
     "  information: mR on clean sets outside the claimed languages, on unknown (U) sets inside and outside",
     "  the claimed languages, and (in",
     "  brackets) on excluded sets, each with the number of datasets. SCT's excluded numbers carry",
     "  \"w/o recovery\": the same number without `scorer.recover_dropped`.",
     "- Full view = all 9 SCT types (a type the tool does not support counts as missed). Shared view =",
     "  EMAIL, PHONE, NUMBER. Speed: CPU, 16 threads, batch 8, 100 FineWeb-2 docs, median of 3; peak",
     "  RAM = adapter VmHWM. No GPU numbers (pinned torch wheels need CUDA 13.2; the box's driver is 12.6).",
     "- Own-table rows are not a ranking (each row is on its own datasets and languages); compare tools",
     "  through the same-set tables.", "")

emit("## Default track (primary)", "")
for view, vname in (("full", "Full view (all 9 types)"), ("shared", "Shared view (EMAIL, PHONE, NUMBER)")):
    for group, gname in (("privacy-synthetic", "privacy-synthetic"), ("privacy-real", "privacy-real (TAB, REDACT)")):
        emit(f"### {vname}, {gname}")
        table(own("default", view, group))
emit("The NER group is in `TABLES.md`: no model entry claims any NER language (ar, th, bn, hi, ta, zh,",
     "ja), so only datafog (no claim) has an in-claim NER score there; SCT's NER cells are D.", "")

emit("### Same-set, default, full view (the non-SCT entries with clean sets)", "",
     "The only comparison between tools in this report. Privacy-synthetic:", "")
table(same("default", "full", "privacy-synthetic"))
emit("Privacy-real:")
table(same("default", "full", "privacy-real"))
emit("Shared view, privacy-real:")
table(same("default", "shared", "privacy-real"))

emit("## Tuned track (secondary)", "",
     "Same tables on the tuned configurations. Read with the caveat above: each tuned configuration",
     "was chosen on the dev parts of all nine privacy sets.", "")
for group in ("privacy-synthetic", "privacy-real"):
    emit(f"### Full view, {group}")
    table(own("tuned", "full", group))
emit("### Same-set, tuned, full view, privacy-real")
table(same("tuned", "full", "privacy-real"))
emit("Privacy-synthetic: " + same("tuned", "full", "privacy-synthetic")[0], "")

emit("## SCT: excluded cells, with and without the recovery rule", "",
     "Every SCT cell is excluded (D or D-family; sct-gliner also U), so nothing here is a comparison.",
     "Per SCT entry and dataset, full view, claimed languages (sct-gliner: en; `°` = no en doc), masked",
     "recall and doc leak with and without `scorer.recover_dropped` (excluded cells in parentheses in",
     "every number column), and the copies it recovered.", "")
for track in ("default", "tuned"):
    emit(f"### {track} track")
    table(report.recovery_table(bundle, track))

emit("## Recovery rule and its audit", "",
     "Why: SCT's spans are placed from the reversible placeholders found in its output. A placeholder a",
     "later cleaning step removed (the link text of `[a@b.se](mailto:a@b.se)`, repeated names) leaves",
     "its source copy without a span, so it was scored as leaked although no character of it is",
     "visible. `scorer.recover_dropped` credits such a copy of a value SCT masked elsewhere in the doc",
     "only when all of these hold (`scorer.dropped_copies`, tested in `tests/test_benchmarks_compare.py`):", "",
     "- the value has 4+ non-space chars and `leak` finds neither it nor any of its words of 4+ chars",
     "  (`LEAK_MIN_CHARS` = 4) in SCT's output;",
     "- no word of it of 2+ chars is a whole word of the output, case- and accent-insensitive (NFKD,",
     "  marks dropped, casefolded);",
     "- its separator-free folded form is not inside the separator-free folded output;",
     "- the copy is a whole token (no letter or digit glued on).", "",
     "Recovered copies also count as predictions, so they can lower SCT's precision. `runner.py",
     "--no-recover` gives the mapping without them (`norecover/`).", "",
     "Audit (`benchmarks/compare/audit_recover.py`, counts in `audit/`): a seeded sample of 50 recovered",
     "copies per SCT entry, dataset and track (all copies when fewer), each checked for any visible",
     "trace with the folded checks above. First pass, with the rule before the folded checks (2+ char",
     f"words, casefolded only): {first_traces} of {first_sampled} sampled copies had a trace (at most",
     f"{max_traces} of 50 in one set, in {len(sets_with_traces)} entry-dataset-track cells), all of them the",
     "value's separator-free form inside a longer token of the output (none was a visible word). Such a",
     "copy may still be readable, so crediting it could be a wrong credit (the gretel ones were extra",
     "predictions, not on gold; REDACT/TAB/openpii ones were not classified, no text). The rule was",
     f"tightened, not dropped: the folded checks are now part of it. Final rule: {passed} of {sampled} sampled copies pass;",
     f"{on_gold} of them overlap a gold span (credited recall), the rest are extra predictions. REDACT and",
     "TAB were audited by counts and pass/fail only; no dataset text was written for them. The gretel",
     "sample was also read by hand (private notes): the credited copies are mostly names, company",
     "names and markdown link text, and none of the values shows in SCT's output in any form.", "")
for track in ("default", "tuned"):
    emit(f"### Audit counts, {track} track")
    table(report.audit_table(bundle, track))

for track in ("default", "tuned"):
    w = bundle / "norecover" / track
    emit(f"## Per dataset ({track}, full view, masked F2, claimed languages only, completed docs)", "",
         "`(x)` = excluded cell (T/E/D, family included), information only; `x?` = unknown training data;",
         "`°` = no doc of the dataset in the entry's claimed languages. SCT cells show the value without",
         "the recovery rule beside it.")
    table(report.dataset_table(bundle / track, AGG, without_dir=w))
    emit(f"### Outside claimed languages ({track}; information only, not a loss; `-` = every doc in a claimed language)")
    table(report.dataset_table(bundle / track, AGG, inside=False, without_dir=w))

emit("## Per language (default, full view, masked recall, clean sets only)", "",
     "Each cell: mean over the entry's clean datasets of the group that have the language. `°` = a",
     "language outside the entry's claimed languages: shown apart from the claim, not a loss. `-` = the",
     "entry has no clean dataset with that language. Tuned, unknown-set and NER tables are in `TABLES.md`.", "")
for group in ("privacy-synthetic", "privacy-real"):
    emit(f"### {group}")
    table(report.language_table(bundle / "default", AGG, group))

emit("## Contamination matrix", "",
     "T = trained on it, E = vendor evaluates on it, D = SCT developed against it in earlier rounds, U =",
     "training data undisclosed, `X-family` = X on another dataset of the same generator family, - =",
     "none known (model cards at the pinned revisions, checked 2026-11-02; families 2026-11-03). Bold =",
     "excluded, italic = unknown, plain = clean.")
table(report.contamination_table(registry.ENTRIES))

NL = chr(10)
STATIC = Path(sys.argv[2]).read_text().replace("COMPLETION_DEFAULT", NL.join(
    [*report.completion_table(bundle / "default", AGG), "", "Tuned track:", "",
     *report.completion_table(bundle / "tuned", AGG)]))
emit(STATIC.rstrip())
print("\n".join(out).rstrip() + "\n")
