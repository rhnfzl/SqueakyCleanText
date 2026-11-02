"""Tool-comparison harness: dev/test split, new loaders' span building, entry tables (no dataset data)."""

from benchmarks.compare import entries, split
from benchmarks.datasets import MODULES, get
from benchmarks.datasets.gretel_en import entity_spans
from benchmarks.labels import DATASET_LABELS, normalise_gold


def rec(rid, group=None):
    r = {"id": rid, "lang": "en", "text": "x", "spans": []}
    if group is not None:
        r["group"] = group
    return r


def test_split_is_disjoint_complete_and_order_free():
    records = [rec(str(i)) for i in range(2000)]
    dev, test = split.split("d", records, "dev"), split.split("d", records, "test")
    assert {r["id"] for r in dev}.isdisjoint(r["id"] for r in test)
    assert len(dev) + len(test) == len(records)
    assert 0.15 < len(dev) / len(records) < 0.25
    assert [r["id"] for r in split.split("d", records[::-1], "dev")] == [r["id"] for r in dev][::-1]
    assert split.split("d", records, "dev", seed=14) != dev  # seed changes the split


def test_split_keeps_a_group_on_one_side():
    records = [rec(f"{g}:{i}", group=f"t{g}") for g in range(50) for i in range(5)]
    dev_groups = {r["group"] for r in split.split("d", records, "dev")}
    assert dev_groups and dev_groups.isdisjoint(r["group"] for r in split.split("d", records, "test"))
    assert all(sum(r["group"] == g for r in split.split("d", records, "dev")) == 5 for g in dev_groups)


def test_split_rejects_unknown_part():
    try:
        split.split("d", [], "train")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_gretel_en_entity_strings_become_every_occurrence():
    text = "Mail ann@x.io or ann@x.io, SSN 123-45-6789."
    spans = entity_spans(text, [{"entity": "ann@x.io", "types": ["email"]},
                                {"entity": "123-45-6789", "types": ["ssn"]}, {"entity": "", "types": ["name"]}])
    assert spans == [{"start": 5, "end": 13, "label": "email"}, {"start": 17, "end": 25, "label": "email"},
                     {"start": 31, "end": 42, "label": "ssn"}]


def test_argilla_name_parts_merge():
    text = "Dear Ann Lee Berg,"
    gold, _ = normalise_gold(text, [{"start": 5, "end": 8, "label": "FIRSTNAME"},
                                    {"start": 9, "end": 12, "label": "MIDDLENAME"},
                                    {"start": 13, "end": 17, "label": "LASTNAME"}], DATASET_LABELS["argilla_pii"])
    assert gold == [{"start": 5, "end": 17, "label": "PERSON"}]


def test_new_datasets_registered_with_pins_and_publish_rules():
    for name in ("gretel_en", "argilla_pii"):
        module = get(name)
        assert name in MODULES and name in DATASET_LABELS
        assert len(module.REVISION) == 40 and module.SUITE == "pii"
    assert get("argilla_pii").PUBLISH == "scores-only"


def test_entries_are_complete_and_map_to_sct_types():
    assert len(entries.ENTRIES) == 8 and {"sct-default", "sct-gliner"} <= set(entries.ENTRIES)
    for name, entry in entries.ENTRIES.items():
        for key in ("package", "checkpoint", "revision", "config", "threshold", "mask", "device", "gpu_gb"):
            assert key in entry, (name, key)
        assert set(entry["labels"].values()) <= set(entries.SCT_TYPES), name
        assert set(entry["supports"]) <= set(entries.SCT_TYPES), name
        assert entry["gpu_gb"] <= 5, name  # shared box: about 6 GB free per GPU


# --- comparison harness: scorer, supervisor, runner (hand-made text and spans only) ---

import json  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402

import pytest  # noqa: E402

from benchmarks.compare import runner, scorer  # noqa: E402

MAP = {"person": "PERSON", "email": "EMAIL", "date": "DATE", "ssn": "NUMBER"}
TEXT = "Ann Lee and Bo Ek mailed ann@x.io on 2020-01-02; SSN 123-45-6789. Ann Lee again."
#       0         1         2         3         4         5         6         7


def span(start, end, label, score=None):
    return {"start": start, "end": end, "label": label, "score": score}


def gold_record(rid="d1", spans=None, lang="en", text=TEXT):
    if spans is None:
        spans = [span(0, 7, "person"), span(12, 17, "person"), span(25, 33, "email"),
                 span(37, 47, "date"), span(53, 64, "ssn"), span(66, 73, "person")]
    return {"id": rid, "lang": lang, "text": text, "spans": [{k: s[k] for k in ("start", "end", "label")} for s in spans]}


def test_map_spans_drops_unmapped_thresholds_and_merges_same_type():
    spans, dropped = scorer.map_spans(
        [span(0, 3, "per", 0.9), span(2, 7, "per", 0.4), span(2, 7, "org", 0.9), span(5, 5, "per", 0.9),
         span(10, 12, "secret", 0.9), span(20, 25, "per", 0.1)], {"per": "PERSON", "org": "ORGANISATION"}, threshold=0.2)
    assert dropped == 1
    assert spans == [span(0, 7, "PERSON", 0.9), span(2, 7, "ORGANISATION", 0.9)]


def test_common_mask_merges_regions_of_any_type():
    assert scorer.mask_text("abcdefghij", [span(1, 4, "PERSON"), span(3, 6, "EMAIL"), span(8, 9, "IP")]) == \
        "a<PERSON>gh<IP>j"
    assert scorer.mask_text("abc", []) == "abc"


def test_one_to_one_partial_and_masked_recall():
    record = gold_record()
    one_long = [span(0, 17, "PERSON")]  # covers both names
    c = scorer.doc_stats(record, one_long, {k: v for k, v in MAP.items() if v != "DATE"})
    assert c["gold"] == 5 and c["pred"] == 1
    assert c["partial_tp"] == 1  # one-to-one: a single prediction matches one gold
    assert c["gold_masked"] == 2  # but it masks both names
    assert c["strict_tp"] == 0


def test_leak_measures_on_the_common_mask():
    record = gold_record()
    mapping = {k: v for k, v in MAP.items() if v != "DATE"}
    preds = [span(0, 7, "PERSON"), span(25, 30, "EMAIL")]  # e-mail half masked; second "Ann Lee" missed
    c = scorer.doc_stats(record, preds, mapping)
    assert c["mentions"] == 5
    assert c["exposed"] == 4  # Bo Ek, e-mail tail, SSN, second Ann Lee
    assert c["complete_exposure"] == 3  # Bo Ek, SSN, second Ann Lee
    assert c["alnum"] == 6 + 4 + 6 + 9 + 6 and c["alnum_survived"] == 4 + 2 + 9 + 6
    assert c["repeated"] == 2 and c["repeated_exposed"] == 2  # both "Ann Lee" mentions: one copy unmasked
    assert c["exact_visible"] == 4  # Bo Ek, SSN and both "Ann Lee" (the value is visible via the second copy)
    assert c["docs_exposed"] == 1 and c["docs_with_gold"] == 1


def test_overmasking_clean_docs_and_char_scores():
    clean = {"id": "c", "lang": "en", "text": "nothing personal here", "spans": []}
    c = scorer.doc_stats(clean, [span(0, 7, "PERSON")], MAP)
    assert c["clean_docs"] == 1 and c["clean_docs_fp"] == 1 and c["nongold_masked"] == 7
    m = scorer.metrics(c)
    assert m["masked_recall"] is None and m["clean_doc_fp_rate"] == 1.0  # no gold: N/A, not 0
    assert m["overmask_rate"] == round(7 / len(clean["text"]), 4)
    record = gold_record(spans=[span(0, 7, "person")])
    c = scorer.doc_stats(record, [span(0, 3, "PERSON"), span(75, 77, "PERSON")], MAP)
    m = scorer.metrics(c)
    assert (m["char_precision"], m["char_recall"]) == (0.6, round(3 / 7, 4))
    assert m["char_f2"] == round(5 * 0.6 * (3 / 7) / (4 * 0.6 + 3 / 7), 4)


def test_perfect_prediction_scores_one_and_chrf_one():
    record = gold_record()
    mapping = {k: v for k, v in MAP.items() if v != "DATE"}
    preds = [span(s["start"], s["end"], MAP[s["label"]]) for s in record["spans"] if s["label"] != "date"]
    m = scorer.metrics(scorer.doc_stats(record, preds, mapping))
    assert m["strict_f1"] == m["partial_f1"] == m["masked_recall"] == m["chrf"] == 1.0
    assert m["char_survival"] == 0.0 and m["doc_leak_rate"] == 0.0 and m["overmask_rate"] == 0.0


def test_macro_and_micro_aggregation():
    big = scorer.doc_stats(gold_record("a", spans=[span(0, 7, "person")]), [span(0, 7, "PERSON")], MAP)
    small = scorer.doc_stats(gold_record("b", spans=[span(0, 7, "person"), span(12, 17, "person")]), [], MAP)
    one = scorer.summarise([("en", big), ("en", big), ("en", big), ("de", small)])
    assert one["by_language"]["en"]["masked_recall"] == 1.0 and one["by_language"]["de"]["masked_recall"] == 0.0
    assert one["macro"]["masked_recall"] == 0.5
    assert one["micro"]["masked_recall"] == 0.6  # 3 of 5 gold
    agg = scorer.aggregate({"x": [("en", big)], "y": [("en", small), ("en", small)]})
    assert agg["macro"]["masked_recall"] == 0.5 and agg["micro"]["masked_recall"] == 0.2


def test_curve_and_operating_point():
    records = [gold_record("a", spans=[span(0, 7, "person"), span(25, 33, "email")])]
    outputs = {"a": {"id": "a", "spans": [span(0, 7, "per", 0.8), span(25, 33, "mail", 0.3), span(40, 44, "per", 0.2)]}}
    rows = scorer.curve(records, outputs, {"per": "PERSON", "mail": "EMAIL"}, MAP)
    by_t = {r["threshold"]: r for r in rows}
    assert by_t[0.05]["masked_recall"] == 1.0 and by_t[0.05]["partial_precision"] == round(2 / 3, 4)
    assert by_t[0.5]["masked_recall"] == 0.5 and by_t[0.9]["masked_recall"] == 0.0
    assert scorer.operating_point(rows, 1.0) == 0.3 and scorer.operating_point(rows, 0.5) == 0.8
    assert scorer.curve(records, {"a": {"id": "a", "spans": [span(0, 7, "per")]}}, {"per": "PERSON"}, MAP) == []


def test_score_run_views_failures_and_effective_protection():
    records = [gold_record("a"), gold_record("b")]
    outputs = {"a": {"id": "a", "spans": [span(25, 33, "EMAIL"), span(53, 64, "NUMBER")], "seconds": 0.1},
               "b": {"id": "b", "spans": [], "error": "timeout: no output for 1s", "seconds": None}}
    labels = {"EMAIL": "EMAIL", "NUMBER": "NUMBER"}
    views = scorer.views_for(("EMAIL", "NUMBER"), ("EMAIL",), ("PERSON", "EMAIL", "NUMBER", "DATE"))
    result, per_doc = scorer.score_run(records, outputs, labels, MAP, views)
    assert result["completion_rate"] == 0.5 and result["flag_low_completion"]
    assert result["failure_kinds"] == {"timeout": 1}
    sup, shared, full = (result["views"][v] for v in ("supported", "shared", "full"))
    assert sup["compare"]["micro"]["masked_recall"] == 1.0
    assert shared["compare"]["micro"]["gold"] == 1
    assert full["compare"]["micro"]["masked_recall"] == 0.4  # 3 persons missed (unsupported counts as missed)
    assert sup["effective"]["masked_recall"] == 0.5  # the failed doc counts as unmasked
    assert full["with_date"]["gold"] == 6
    assert "partial" in sup["pii"] and "leak" in sup["pii"]
    assert [x[0] for x in per_doc["shared"]] == ["a", "b"]
    assert result["curve"] == [] and result["native"] is None


def test_paired_bootstrap_and_mcnemar():
    good = [(f"d{i}", f"g{i}", "en", scorer.doc_stats(gold_record(f"d{i}", spans=[span(0, 7, "person")]),
                                                        [span(0, 7, "PERSON")], MAP)) for i in range(30)]
    bad = [(i, g, lang, scorer.doc_stats(gold_record(i, spans=[span(0, 7, "person")]), [], MAP)) for i, g, lang, _ in good]
    same = scorer.paired_bootstrap(good, good, samples=200)
    assert same["metrics"]["masked_recall"]["diff_ci"] == [0.0, 0.0] and not same["metrics"]["masked_recall"]["excludes_zero"]
    diff = scorer.paired_bootstrap(good, bad, samples=200)
    assert diff["metrics"]["masked_recall"]["diff_ci"] == [1.0, 1.0] and diff["metrics"]["masked_recall"]["excludes_zero"]
    assert diff["doc_leak_mcnemar"]["only_b"] == 30 and diff["doc_leak_mcnemar"]["p_value"] < 1e-6
    with pytest.raises(ValueError):
        scorer.paired_bootstrap(good, bad[::-1], samples=10)
    assert scorer.mcnemar([True, False], [True, False])["p_value"] == 1.0


DUMMY = [sys.executable, str(runner.ADAPTERS / "dummy_adapter.py")]


def dummy_cmd(config):
    return [*DUMMY, "--config", json.dumps(config)]


def docs(n):
    return [{"id": f"d{i}", "lang": "en", "text": f"doc {i} mail a{i}@x.io"} for i in range(n)]


def test_supervisor_keeps_errors_and_bad_spans(tmp_path):
    results, info = runner.supervise(dummy_cmd({"raise": ["d1"], "bad": ["d2"]}), docs(4), tmp_path,
                                     runner.child_env(), doc_timeout=30, start_timeout=30)
    assert set(results) == {"d0", "d1", "d2", "d3"} and info["processes"] == 1
    assert results["d1"]["error"].startswith("RuntimeError") and results["d1"]["spans"] == []
    assert results["d2"]["error"].startswith("ValueError: span offsets out of range")
    assert results["d0"]["spans"] == [span(11, 18, "EMAIL", 0.9), span(4, 5, "NUMBER", 0.6)]
    assert not list(tmp_path.glob("in-*.jsonl"))  # inputs (dataset text) are removed


def test_supervisor_retries_crash_then_records_timeouts(tmp_path):
    config = {"crash_once": ["d1"], "state_dir": str(tmp_path), "crash": ["d3"], "sleep": {"d4": 5}}
    results, info = runner.supervise(dummy_cmd(config), docs(6), tmp_path / "w", runner.child_env(),
                                     doc_timeout=1.0, start_timeout=10, retries=1)
    assert "error" not in results["d1"]  # crashed once, the retry worked
    assert results["d3"]["error"] == "crash: exit 3"
    assert results["d4"]["error"].startswith("timeout")
    assert "error" not in results["d5"] and "error" not in results["d0"]
    assert info["restarts"]["crash"] == 3 and info["restarts"]["timeout"] == 2


def test_supervisor_aborts_when_the_adapter_cannot_start(tmp_path):
    cmd = [sys.executable, "-c", "import sys; sys.exit(2)"]
    results, info = runner.supervise(cmd, docs(10), tmp_path, runner.child_env(), doc_timeout=5, start_timeout=5)
    assert len(results) == 10 and all("error" in r for r in results.values())
    assert results["d9"]["error"].startswith("aborted") and info["processes"] == 2 * runner.MAX_FAILED_STREAK


def test_offset_fixtures_pass_for_a_correct_adapter(tmp_path):
    fixtures = [{"id": k, "lang": lang, "text": text} for k, (lang, text) in runner.OFFSET_FIXTURES.items()]
    results, _ = runner.supervise(dummy_cmd({}), fixtures, tmp_path, runner.child_env(), 30, 30)
    assert runner.check_offsets(results, {"EMAIL": "EMAIL"}) == ({}, {})
    shifted = {k: {**v, "spans": [{**s, "start": s["start"] + 1} for s in v["spans"]]} for k, v in results.items()}
    problems, misses = runner.check_offsets(shifted, {"EMAIL": "EMAIL"})
    assert set(problems) == set(runner.OFFSET_FIXTURES) and misses == {}


def test_offset_check_separates_detection_errors_from_offset_bugs():
    text = runner.OFFSET_FIXTURES["cjk"][1]
    start = text.index(runner.FIXTURE_EMAIL)
    end = start + len(runner.FIXTURE_EMAIL)
    labels = {"EMAIL": "EMAIL"}

    def check(spans):  # -> (cjk is an offset problem, cjk is a detection error)
        problems, misses = runner.check_offsets({"cjk": {"spans": [span(a, b, "EMAIL") for a, b in spans]}}, labels)
        return "cjk" in problems, "cjk" in misses

    assert check([(start, end)]) == (False, False)
    assert check([(start, end + 3)]) == (False, True)  # swallowed the next word: detection error
    assert check([]) == (False, True)  # missed: detection error
    assert check([(start + 2, end + 2)]) == (True, False)  # misplaced: offset bug
    assert check([(start, end - 1)]) == (True, False)  # cut short: offset bug
    assert "latin-combining" in runner.check_offsets({}, labels)[0]  # no output at all


def test_runner_end_to_end_with_dummy(tmp_path, monkeypatch):
    records = [gold_record("a", spans=[span(25, 33, "email"), span(53, 64, "ssn")]),
               gold_record("b", spans=[span(0, 7, "name")]),
               {"id": "c", "lang": "en", "text": "no pii at all", "spans": []}]
    monkeypatch.setattr(runner.split, "load", lambda dataset, part, n, seed: records)
    out = tmp_path / "out"
    runner.main(["--entries", "dummy", "--datasets", "gretel_en", "--part", "dev", "--out", str(out),
                 "--cache", str(tmp_path / "cache"), "--doc-timeout", "30", "--start-timeout", "30"])
    result = json.loads((out / "dummy" / "gretel_en-dev.json").read_text())
    assert result["completion_rate"] == 1.0 and result["contamination"] == "-"
    assert result["views"]["supported"]["compare"]["micro"]["masked_recall"] == 1.0
    assert result["views"]["full"]["compare"]["micro"]["masked_recall"] == round(2 / 3, 4)
    assert result["curve"] and json.loads((out / "dummy" / "offset-check.json").read_text())["passed"]
    preds = [json.loads(line) for line in (out / "predictions" / "dummy" / "gretel_en-dev.jsonl").read_text().splitlines()]
    assert all(set(p) == {"id", "start", "end", "label", "score"} for p in preds)
    for path in out.rglob("*"):  # no dataset text in the result bundle
        if path.is_file():
            assert "no pii at all" not in path.read_text() and "Ann Lee" not in path.read_text()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["aggregate"]["dummy"]["full"]["privacy-synthetic"]["datasets"] == ["gretel_en"]


def test_registry_has_pinned_envs_and_contamination():
    from benchmarks.compare import entries as reg

    for name in reg.ENTRIES:
        t = reg.tool(name)
        assert t["requirements"].exists(), name
        pins = [line for line in t["requirements"].read_text().splitlines() if line and not line.startswith("#")]
        assert pins and all("==" in p or " @ https://" in p for p in pins), name
        assert reg.contamination(name, "gretel_en") in {"-", "U", "E", "T"}
    assert reg.tool("sct-default")["adapter"] == "sct_adapter.py" and reg.tool("dummy")["requirements"] is None
    assert reg.contamination("gliner-nvidia", "nemotron") == "T" and reg.contamination("sct-default", "redact") == "D"
    with pytest.raises(ValueError):
        reg.tool("nope")
    assert Counter(reg.contamination("opf", d) for d in ("tab", "openpii")) == Counter({"U": 2})


def test_speed_mode_reports_cold_start_latency_throughput_and_rss(tmp_path):
    from benchmarks.compare import speed

    run = speed.measure(dummy_cmd({}), docs(30), tmp_path, runner.child_env(), batch_size=4, timeout=60, poll_vram=False)
    assert "error" not in run and run["cold_start_seconds"] > 0
    assert run["latency"]["docs"] == 30 and run["latency"]["p95_ms"] >= run["latency"]["median_ms"] >= 0
    assert sorted(t["batch_size"] for t in run["throughput"]) == [4, 8]
    assert run["done"]["peak_rss_mb"] > 1
    summary = speed.summarise([run, run, {"error": "timeout after 1s"}])
    assert summary["repeats"] == 3 and summary["failed"] == ["timeout after 1s"]
    assert summary["common_batch_docs_per_second"]["median"] > 0 and len(summary["other_batch"]) == 2


def _adapter_module(name):
    import importlib.util

    sys.path.insert(0, str(runner.ADAPTERS))
    try:
        spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), runner.ADAPTERS / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # stdlib-only at import time: tool packages load inside load()
        return module
    finally:
        sys.path.remove(str(runner.ADAPTERS))


def test_every_entry_has_a_runnable_adapter():
    from benchmarks.compare import entries as reg

    for name in reg.ENTRIES:
        adapter = reg.tool(name)["adapter"]
        assert adapter and adapter.endswith("_adapter.py") and (runner.ADAPTERS / adapter).exists(), name
        assert callable(_adapter_module(adapter).load), name


def test_adapter_label_tables_match_the_frozen_entries():
    from benchmarks.compare import entries as reg

    bardsai = _adapter_module("bardsai_v3_adapter.py")
    assert reg.ENTRIES["bardsai-v3"]["labels"] == bardsai.LABELS
    assert not set(bardsai.OUT_OF_SCOPE) & set(bardsai.LABELS)
    nvidia = _adapter_module("gliner_nvidia_adapter.py")
    assert nvidia.COMMON == list(reg.GLINER_PROMPTS)  # the adapter's literal copy of the shared prompts
    assert set(nvidia.CARD) <= set(reg.ENTRIES["gliner-nvidia"]["labels"])
    fastino = _adapter_module("gliner2_fastino_adapter.py")
    assert {k: v for k, v in fastino.CARD_LABELS.items() if v} == reg.FASTINO_CARD
    for name in ("gliner-nvidia", "gliner2-fastino"):  # card names never re-map a shared prompt
        labels = reg.ENTRIES[name]["labels"]
        assert all(labels[k] == v for k, v in reg.GLINER_PROMPTS.items()), name


# --- Codex review round 1 fixes (C1-C13) ---

def test_c1_full_miss_scores_f1_zero_not_na_in_the_macro():
    record = gold_record(spans=[span(0, 7, "person")])
    hit = scorer.doc_stats(record, [span(0, 7, "PERSON")], MAP)
    miss = scorer.doc_stats(record, [], MAP)
    m = scorer.metrics(miss)
    assert m["strict_f1"] == m["partial_f1"] == m["char_f2"] == 0.0
    assert scorer.summarise([("en", hit), ("de", miss)])["macro"]["strict_f1"] == 0.5
    assert scorer.metrics(scorer.doc_stats({"id": "c", "lang": "en", "text": "x", "spans": []}, [], MAP))["strict_f1"] is None


def test_c2_masking_metrics_share_one_type_agnostic_prediction_set():
    record = gold_record(spans=[span(0, 7, "person"), span(53, 64, "ssn")])
    person_only = {"person": "PERSON"}
    c = scorer.doc_stats(record, [span(0, 7, "NUMBER"), span(53, 64, "NUMBER")], person_only)
    assert c["gold_masked"] == 1 and c["exposed"] == 0 and c["docs_exposed"] == 0  # masked, and not leaked
    assert c["char_tp"] == 7 and c["char_pred"] == 7  # the NUMBER on out-of-view SSN gold is no char FP
    assert c["strict_tp"] == 0  # typed scores still need the type


def test_c3_datafog_accepts_the_runners_runtime_keys():
    datafog = _adapter_module("datafog_adapter.py")
    assert datafog.settings({"threads": 4, "device": "cpu", "gpu_mem_gb": 5.0}) == datafog.DEFAULTS
    with pytest.raises(ValueError):
        datafog.settings({"bogus": 1})


def test_c4_curve_never_reports_points_below_the_run_threshold():
    records = [gold_record("a", spans=[span(0, 7, "person")])]
    outputs = {"a": {"id": "a", "spans": [span(0, 7, "per", 0.8)]}}
    rows = scorer.curve(records, outputs, {"per": "PERSON"}, MAP, floor=0.5)
    assert min(r["threshold"] for r in rows) == 0.5
    views = scorer.views_for(("PERSON",), ("PERSON",), ("PERSON",))
    result, _ = scorer.score_run(records, outputs, {"per": "PERSON"}, MAP, views, curve_by_filter=False)
    assert result["curve"] is None  # OPF: its score is no operating point
    assert runner.curve_floor({}, entries.ENTRIES["gliner-nvidia"]) == 0.5
    assert runner.curve_floor({"threshold": 0.05}, entries.ENTRIES["gliner-nvidia"]) == 0.05
    assert runner.curve_floor({}, entries.ENTRIES["presidio"]) is None
    assert entries.ENTRIES["opf"]["curve_by_filter"] is False


def test_c5_reuse_needs_a_matching_cache_key(tmp_path, monkeypatch):
    records = [gold_record("a", spans=[span(25, 33, "email")])]
    key = runner.cache_key({"t": 1}, "dummy_adapter.py", "freeze", records)
    assert key == runner.cache_key({"t": 1}, "dummy_adapter.py", "freeze", records)
    assert key != runner.cache_key({"t": 2}, "dummy_adapter.py", "freeze", records)
    assert key != runner.cache_key({"t": 1}, "dummy_adapter.py", "freeze2", records)
    assert key != runner.cache_key({"t": 1}, "dummy_adapter.py", "freeze", [{**records[0], "text": TEXT + "!"}])
    monkeypatch.setattr(runner.split, "load", lambda dataset, part, n, seed: records)
    args = ["--entries", "dummy", "--datasets", "gretel_en", "--part", "dev", "--cache", str(tmp_path / "cache"),
            "--doc-timeout", "30", "--start-timeout", "30", "--skip-offset-check", "--reuse"]
    runner.main([*args, "--out", str(tmp_path / "one")])
    runner.main([*args, "--out", str(tmp_path / "two"), "--config", 'dummy={"raise": ["a"]}'])
    two = json.loads((tmp_path / "two" / "dummy" / "gretel_en-dev.json").read_text())
    assert two["completion_rate"] == 0.0 and "reused" not in two["process"]  # new config: rerun, not reused
    runner.main([*args, "--out", str(tmp_path / "three"), "--config", 'dummy={"raise": ["a"]}'])
    assert "reused" in json.loads((tmp_path / "three" / "dummy" / "gretel_en-dev.json").read_text())["process"]


def test_c6_sct_offsets_survive_entity_decoding_and_invisible_chars():
    from sct.utils.normtext import NormaliseText, unescape_markup

    from scripts.benchmark_release import predicted_spans

    sct = _adapter_module("sct_adapter.py")
    fix = NormaliseText().fix_bad_unicode

    def normalise(token):
        return unescape_markup(fix(token))

    for text in ("Mail ana&#64;example.org now", "Mail ana​@example.org now"):
        view = sct.sct_view(text, normalise)
        assert view[0] == "Mail ana@example.org now"
        got = predicted_spans(text, "Mail <EMAIL_0> now", {"<EMAIL_0>": "ana@example.org"}, view)
        assert [(s.start, s.end) for s in got] == [(5, len(text) - 4)]
    text = "Write ana&#64;example.org then ana@example.org ok"
    got = predicted_spans(text, "Write <EMAIL_0> then <EMAIL_1> ok",
                          {"<EMAIL_0>": "ana@example.org", "<EMAIL_1>": "ana@example.org"}, sct.sct_view(text, normalise))
    assert [text[s.start:s.end] for s in got] == ["ana&#64;example.org", "ana@example.org"]


def test_c7_nvidia_postcode_is_a_number_like_gold_and_fastino():
    assert entries.NVIDIA_CARD["postcode"] == DATASET_LABELS["nemotron"]["postcode"] == entries.FASTINO_CARD["postal_code"]


def test_c8_sct_year_maps_to_date_before_scope_filtering():
    for name in ("sct-default", "sct-gliner"):
        spans, dropped = scorer.map_spans([span(0, 4, "YEAR"), span(5, 9, "IBAN")], entries.ENTRIES[name]["labels"])
        assert dropped == 0 and [s["label"] for s in spans] == ["DATE", "NUMBER"]


def test_c9_one_language_track_for_every_entry():
    gold = {name: entries.tool(name, "gold")["adapter_config"] for name in entries.ENTRIES}
    native = {name: entries.tool(name, "native")["adapter_config"] for name in entries.ENTRIES}
    assert gold["sct-default"]["language"] == gold["presidio"]["language"] == "gold" and gold["datafog"]["locales"] == "gold"
    assert native["sct-default"]["language"] == "auto" and "language" not in native["presidio"]
    assert "locales" not in native["datafog"]
    with pytest.raises(ValueError):
        entries.tool("presidio", "mixed")


def test_c10_tab_masking_recall_uses_every_annotator_and_full_coverage():
    record = gold_record(spans=[span(0, 7, "person")])
    record["annotators"] = {"a1": [{"start": 0, "end": 7, "label": "PERSON", "identifier": "DIRECT"}],
                            "a2": [{"start": 12, "end": 17, "label": "PERSON", "identifier": "QUASI"}]}
    outputs = {"d1": {"id": "d1", "spans": [span(0, 3, "PERSON"), span(12, 17, "PERSON")]}}
    views = scorer.views_for(("PERSON",), ("PERSON",), ("PERSON",))
    result, _ = scorer.score_run([record], outputs, {"PERSON": "PERSON"}, MAP, views)
    tab = result["tab_masking"]
    assert tab["annotators"] == 2 and tab["direct"]["micro_recall"] == 0.0 and tab["quasi"]["micro_recall"] == 1.0
    assert result["views"]["shared"]["compare"]["micro"]["masked_recall"] == 1.0  # overlap credit, reported apart


def test_c12_stale_ready_marker_does_not_skip_the_load_allowance(tmp_path):
    slow = tmp_path / "slow.py"
    slow.write_text("import sys, time, json\nargs = sys.argv\ntime.sleep(1.0)\n"
                    "out = args[args.index('--output') + 1]\n"
                    "docs = [json.loads(x) for x in open(args[args.index('--input') + 1])]\n"
                    "open(out, 'w').write(''.join(json.dumps({'id': d['id'], 'spans': []}) + '\\n' for d in docs))\n")
    work = tmp_path / "w"
    work.mkdir()
    (work / "out-1.jsonl.ready").write_text("1")  # left by an earlier run
    results, info = runner.supervise([sys.executable, str(slow)], docs(1), work, runner.child_env(),
                                     doc_timeout=0.3, start_timeout=30)
    assert "error" not in results["d0"] and info["processes"] == 1


def test_c13_bootstrap_ci_describes_the_language_macro():
    def doc(i, lang, hit):
        preds = [span(0, 7, "PERSON")] if hit else []
        c = scorer.doc_stats(gold_record(f"{lang}{i}", spans=[span(0, 7, "person")]), preds, MAP)
        return (f"{lang}{i}", f"{lang}{i}", lang, c)

    a = [doc(i, "en", True) for i in range(20)] + [doc(0, "de", False)]
    b = [doc(i, "en", False) for i in range(20)] + [doc(0, "de", True)]
    out = scorer.paired_bootstrap(a, b, samples=200)["metrics"]["masked_recall"]
    assert out["a"] == out["b"] == 0.5  # macro, not micro (20/21 vs 1/21)
    assert out["diff_ci"][0] <= 0 <= out["diff_ci"][1] and not out["excludes_zero"]
