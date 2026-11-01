import json
from collections import Counter

from scripts.build_multilingual_eval import LABELS, OUTPUT, build, fill, to_jsonl
from scripts.run_multilingual_eval import to_markdown
from sct.evaluation import load_evaluation_jsonl

RECORDS = [json.loads(line) for line in OUTPUT.read_text(encoding="utf-8").splitlines()]


def test_committed_file_matches_builder():
    assert OUTPUT.read_text(encoding="utf-8") == to_jsonl(build())


def test_fill_spans_cover_inserted_values():
    text, entities = fill("{PERSON}去年搬到{LOCATION}。", {"PERSON": "张伟", "LOCATION": "上海"})
    assert text == "张伟去年搬到上海。"
    assert [text[e["start"]:e["end"]] for e in entities] == ["张伟", "上海"]


def test_spans_valid_and_non_overlapping():
    for record in RECORDS:
        spans = sorted((e["start"], e["end"]) for e in record["entities"])
        assert spans, record["id"]
        for (start, end), (next_start, _) in zip(spans, spans[1:] + [(len(record["text"]), 0)]):
            assert 0 <= start < end <= next_start, record["id"]
            assert record["text"][start:end].strip() == record["text"][start:end], record["id"]
        assert {e["label"] for e in record["entities"]} <= set(LABELS)


def test_language_coverage_and_unique_ids():
    per_language = Counter(record["language"] for record in RECORDS)
    assert len(per_language) >= 10
    assert min(per_language.values()) >= 15
    assert {"hi", "bn", "ta", "zh", "ja", "th", "ar"} <= set(per_language)
    assert len({record["id"] for record in RECORDS}) == len(RECORDS)
    assert any("code-switch" in record["slices"] for record in RECORDS)
    assert len(load_evaluation_jsonl(OUTPUT)) == len(RECORDS)


def test_markdown_table_has_language_and_label_rows():
    metrics = {"recall": 0.5, "precision": 1.0}
    report = {"overall": metrics, "by_language": {"hi": metrics}, "by_label": {"PERSON": metrics}}
    table = to_markdown({"models": {"m": {"report": report}}})
    assert "| hi | 0.50 / 1.00 |" in table
    assert "| PERSON | 0.50 / 1.00 |" in table
    assert "| EMAIL | - |" in table
