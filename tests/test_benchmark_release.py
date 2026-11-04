from benchmarks.score import predicted_spans
from scripts.benchmark_release import _git_commit, find_leaks
from sct.evaluation import EntitySpan, EvaluationExample
import builtins
import importlib
import subprocess
import sys


def test_digits_in_a_user_label_are_a_placeholder():
    # Labels may hold digits (sct.config._USER_LABEL): the scorers must read such tags too.
    assert predicted_spans("Ref AB-1 here", "Ref <EU1_0> here", {"<EU1_0>": "AB-1"}) == [EntitySpan("EU1", 4, 8)]
    example = EvaluationExample(
        id="doc", text="EU1 Smith", language="en", domain="test", expected=(EntitySpan("PERSON", 0, 9),))
    assert find_leaks(example, "<EU1> here") == ([], [])


def test_predicted_spans_maps_placeholders_in_order():
    source = "Call 555 1234, then 555 1234 again; Ann knows."
    output = "Call <NUMBER_0>, then <NUMBER_0> again; <PERSON_0> knows. <EMAIL_0>"
    originals = {"<NUMBER_0>": "555 1234", "<PERSON_0>": "Ann", "<EMAIL_0>": "gone@x.io"}

    assert predicted_spans(source, output, originals) == [
        EntitySpan("NUMBER", 5, 13),
        EntitySpan("NUMBER", 20, 28),
        EntitySpan("PERSON", 36, 39),
        EntitySpan("EMAIL", -1, -1),
    ]


def test_find_leaks_reports_full_and_partial_pii_only():
    text = "Bram de Vries paid 2023 from 8.8.8.8"
    example = EvaluationExample(
        id="doc",
        text=text,
        language="nl",
        domain="test",
        expected=(
            EntitySpan("PERSON", 0, 13),
            EntitySpan("YEAR", 19, 23),
            EntitySpan("IP", 29, 36),
        ),
    )

    leaks, partial = find_leaks(example, "<PERSON> Vries paid 2023 from 8.8.8.8")

    assert leaks == [{"id": "doc", "label": "IP", "value": "8.8.8.8"}]
    assert partial == [{"id": "doc", "label": "PERSON", "value": "Bram de Vries"}]


def test_predicted_spans_skip_untouched_earlier_copy():
    assert predicted_spans("Ann met Ann", "Ann met <PERSON_0>", {"<PERSON_0>": "Ann"}) == [
        EntitySpan("PERSON", 8, 11),
    ]


def test_predicted_spans_map_legacy_type_names_to_dataset_label():
    assert predicted_spans("at 8.8.8.8", "at <IP_ADDRESS_0>", {"<IP_ADDRESS_0>": "8.8.8.8"}) == [
        EntitySpan("IP", 3, 10),
    ]
    assert predicted_spans("t 112", "t <PHONE_NUMBER_0>", {"<PHONE_NUMBER_0>": "112"}) == [
        EntitySpan("PHONE", 2, 5),
    ]


def test_find_leaks_catches_split_values():
    text = "Call 9876543210 from 192.168.1.10"
    example = EvaluationExample(
        id="doc", text=text, language="en", domain="test",
        expected=(EntitySpan("PHONE", 5, 15), EntitySpan("IP", 21, 33)),
    )

    leaks, partial = find_leaks(example, "Call 98765<NUMBER> from <NUMBER_0>.1.10")

    assert leaks == []
    assert [item["label"] for item in partial] == ["PHONE", "IP"]


def test_predicted_spans_context_changed_by_cleanup():
    # Isolated-letter removal drops "x", so the output context is not in the source.
    assert predicted_spans("Ann x met Ann", "Ann met <PERSON_0>", {"<PERSON_0>": "Ann"}) == [
        EntitySpan("PERSON", 10, 13),
    ]


def test_run_batch_reports_throughput_keys(tmp_path):
    from scripts.benchmark_release import ROOT, run_batch
    from tests.conftest import TEST_NER_MODELS

    lines = (ROOT / "evaluation" / "release_benchmark.jsonl").read_text(encoding="utf-8").splitlines()
    dataset = tmp_path / "two.jsonl"
    dataset.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")

    result = run_batch(
        dataset, documents=4, batch_size=2, ner_batch_size=8, intra_op_threads=1,
        repeats=1, long=False, ner_models=TEST_NER_MODELS,
    )

    assert {"settings", "seconds", "docs_per_second", "peak_rss_mb", "output_sha256"} <= set(result)
    assert result["settings"]["documents"] == 4 and len(result["seconds"]) == 1


def test_predicted_spans_map_unicode_folded_value_to_source_offsets():
    # SCT folds full-width text before replacing it.
    source = "Mail Ａｎｎａ＠ｅｘａｍｐｌｅ．ｃｏｍ ok"
    assert predicted_spans(source, "Mail <EMAIL_0> ok", {"<EMAIL_0>": "Anna@example.com"}) == [
        EntitySpan("EMAIL", 5, 21)]
    # A combining accent stays with its letter; a ligature maps to its one source char.
    assert predicted_spans("René met ﬁ", "<PERSON_0> met <X_0>", {"<PERSON_0>": "René", "<X_0>": "fi"}) == [
        EntitySpan("PERSON", 0, 5), EntitySpan("X", 10, 11)]


def test_benchmark_imports_without_resource_module(monkeypatch):
    real_import = builtins.__import__

    def no_resource(name, *args, **kwargs):
        if name == "resource":
            raise ImportError("No module named 'resource'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "scripts.benchmark_release", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_resource)
    module = importlib.import_module("scripts.benchmark_release")
    assert module._peak_rss_mb() is None


def test_git_commit_is_none_when_git_hangs(monkeypatch):
    def hang(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", hang)
    assert _git_commit() is None
