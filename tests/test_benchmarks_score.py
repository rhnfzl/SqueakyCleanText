"""Scorer, label mapping and loader helpers of benchmarks/ (hand-made spans, no dataset data)."""

import fnmatch
import json
import random
import tomllib
from pathlib import Path

import pytest

from benchmarks.datasets import bio_to_spans, sample_per_lang, valid_spans
from benchmarks.labels import DATASET_LABELS, normalise_gold
from benchmarks.run import summary
from benchmarks.score import (
    dominant_script, inject_noise, leak, masking_recall, prepare, score_lid, score_noise, score_pii,
)

ROOT = Path(__file__).resolve().parents[1]
MAP = {"GIVENNAME": "PERSON", "SURNAME": "PERSON", "EMAIL": "EMAIL", "CITY": "LOCATION", "DATE": "DATE"}


def span(start, end, label):
    return {"start": start, "end": end, "label": label}


def record(text, spans, rid="d1", lang="en"):
    return {"id": rid, "lang": lang, "text": text, "spans": spans}


def test_given_and_surname_merge_into_one_person():
    text = "Hi Anna Berg, Bob"
    gold, out = normalise_gold(text, [span(3, 7, "GIVENNAME"), span(8, 12, "SURNAME"), span(14, 17, "GIVENNAME")], MAP)
    assert gold == [span(3, 12, "PERSON"), span(14, 17, "PERSON")]  # ", " between names: no merge
    assert out == []


def test_nested_and_duplicate_same_type_spans_collapse():
    text = "Anna Berg"
    mapping = {"Full": "PERSON", "First": "PERSON", "Last": "PERSON"}
    gold, _ = normalise_gold(text, [span(0, 9, "Full"), span(0, 4, "First"), span(5, 9, "Last"), span(0, 9, "Full")],
                             mapping)
    assert gold == [span(0, 9, "PERSON")]


def test_unmapped_gold_is_out_of_scope_and_does_not_make_false_positives():
    text = "Anna bought a Volvo in Oslo"
    rec = record(text, [span(0, 4, "GIVENNAME"), span(14, 19, "PRODUCT"), span(23, 27, "CITY")])
    preds = [span(0, 4, "PERSON"), span(14, 19, "ORGANISATION"), span(23, 27, "LOCATION"), span(5, 11, "MISC")]
    gold, kept = prepare(rec, preds, MAP)
    assert [g["label"] for g in gold] == ["PERSON", "LOCATION"]
    # ORG on the out-of-scope product is ignored; MISC is not a scored type.
    assert kept == [span(0, 4, "PERSON"), span(23, 27, "LOCATION")]


def test_year_prediction_counts_as_date():
    rec = record("born 1990", [span(5, 9, "DATE")])
    _, kept = prepare(rec, [span(5, 9, "YEAR")], MAP)
    assert kept == [span(5, 9, "DATE")]


def test_strict_and_partial_matching():
    text = "Mail anna@x.org to Anna Berg in Oslo"
    rec = record(text, [span(5, 15, "EMAIL"), span(19, 23, "GIVENNAME"), span(24, 28, "SURNAME"), span(32, 36, "CITY")])
    preds = {"d1": [span(5, 15, "EMAIL"), span(19, 23, "PERSON"), span(32, 36, "PERSON")]}
    out = {"d1": "Mail <EMAIL_0> to <PERSON_0> Berg in <PERSON_1>"}
    result = score_pii([rec], preds, out, MAP)
    strict = result["strict"]["overall"]
    assert (strict["true_positives"], strict["false_positives"], strict["false_negatives"]) == (1, 2, 2)
    partial = result["partial"]["overall"]
    # EMAIL exact, PERSON overlaps the merged name; Oslo tagged PERSON is a type miss.
    assert (partial["gold"], partial["gold_found"], partial["pred"], partial["pred_correct"]) == (3, 2, 3, 2)
    assert partial["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert result["partial"]["by_label"]["LOCATION"]["recall"] == 0.0
    assert result["by_position"]["0-1000"]["gold"] == 3


def test_unfound_prediction_is_a_false_positive():
    rec = record("Anna", [span(0, 4, "GIVENNAME")])
    result = score_pii([rec], {"d1": [span(-1, -1, "PERSON"), span(0, 4, "DATE")]}, {"d1": "<DATE_0>"}, MAP)
    assert result["strict"]["overall"]["false_positives"] == 2
    assert result["partial"]["overall"]["pred_correct"] == 0
    # A type only predicted (no gold) still has every count key.
    assert result["partial"]["by_label"]["DATE"]["gold"] == 0


def test_leak_full_partial_and_placeholder_text():
    assert leak("Anna Berg", "Hi Anna Berg") == "full"
    assert leak("Anna Berg", "Hi <PERSON_0> Berg") == "partial"
    assert leak("Anna Berg", "Hi <PERSON_0>") is None
    assert leak("Al Bo", "Hi <PERSON_0>, Al") is None  # words under 4 chars
    assert leak("PERSON", "<PERSON_0>") is None  # placeholder names are not leaks


def test_leak_rate_excludes_dates_from_total():
    text = "Anna on 2020-01-01"
    rec = record(text, [span(0, 4, "GIVENNAME"), span(8, 18, "DATE")])
    result = score_pii([rec], {"d1": []}, {"d1": text}, MAP)
    assert result["leak"]["overall"] == {"total": 1, "full": 1, "partial": 0, "leak_rate": 1.0,
                                         "full_leak_rate": 1.0}
    assert result["leak"]["by_label"]["DATE"]["total"] == 1


def test_bio_rebuild_offsets():
    text, spans = bio_to_spans(["Anna", "Berg", "lives", "in", "New", "York"],
                               ["B-PER", "I-PER", "O", "O", " B-LOC", "I-LOC"])
    assert text == "Anna Berg lives in New York"
    assert [(text[s["start"]:s["end"]], s["label"]) for s in spans] == [("Anna Berg", "PER"), ("New York", "LOC")]


def test_bio_rebuild_no_separator_and_stray_inside_tag():
    text, spans = bio_to_spans(["北", "京", "和", "上", "海"], ["B-LOC", "I-LOC", "O", "I-LOC", "I-ORG"], sep="")
    assert text == "北京和上海"
    assert [(text[s["start"]:s["end"]], s["label"]) for s in spans] == [("北京", "LOC"), ("上", "LOC"), ("海", "ORG")]


def test_valid_spans_drops_bad_offsets():
    assert valid_spans("abc", [span(0, 3, "X"), span(2, 2, "X"), span(1, 9, "X"), span(-1, 1, "X")]) == [span(0, 3, "X")]


def test_sample_per_lang_is_seeded_and_per_language():
    rows = [{"id": str(i), "lang": "ab"[i % 2]} for i in range(200)]
    first = sample_per_lang(rows, 5, seed=1)
    assert first == sample_per_lang(rows, 5, seed=1)
    assert first != sample_per_lang(rows, 5, seed=2)
    assert [r["lang"] for r in first].count("a") == 5
    # Language "a" keeps its sample when language "b" changes.
    only_a = sample_per_lang([r for r in rows if r["lang"] == "a"], 5, seed=1)
    assert [r for r in first if r["lang"] == "a"] == only_a
    assert [int(r["id"]) for r in first] == sorted(int(r["id"]) for r in first)
    assert len(sample_per_lang(rows, None, seed=1)) == 200


def test_masking_recall_per_annotator():
    text = "Mr Anna Berg, case 123/45, Oslo"
    rec = record(text, [], rid="t1")
    rec["annotators"] = {
        "a1": [{"start": 3, "end": 12, "label": "PERSON", "identifier": "DIRECT"},
               {"start": 19, "end": 25, "label": "CODE", "identifier": "DIRECT"}],
        "a2": [{"start": 27, "end": 31, "label": "LOC", "identifier": "QUASI"},
               {"start": 0, "end": 2, "label": "DEM", "identifier": "NO_MASK"}],
    }
    preds = {"t1": [span(3, 12, "PERSON"), span(19, 22, "NUMBER"), span(23, 25, "NUMBER")]}
    result = masking_recall([rec], preds)
    assert result["direct"]["micro_recall"] == 1.0  # "/" left visible still counts as masked
    assert result["quasi"]["micro_recall"] == 0.0
    assert result["all"]["macro_recall"] == 0.5  # a1: 2/2, a2: 0/1
    assert result["annotators"] == 2


def test_score_lid_in_set_and_out_of_set():
    recs = [record("Hallo wereld", [], "1", "nld"), record("Hello world", [], "2", "eng"),
            record("Grüezi mitenand", [], "3", "als")]
    result = score_lid(recs, {"1": "nld", "2": "deu", "3": "deu"}, {"nld", "eng", "deu"})
    assert result["accuracy"] == 0.5
    assert result["top_confusions"] == {"eng->deu": 1}
    assert result["out_of_set"] == {"documents": 1, "answered": 1, "false_positive_rate": 1.0}
    assert result["by_script"]["LATIN"]["total"] == 2


def test_dominant_script():
    assert dominant_script("Привет, world") == "CYRILLIC"
    assert dominant_script("123 !!") == "NONE"


def test_noise_injection_and_removal_scoring():
    text, items = inject_noise("Un café au lait", random.Random(1))  # noqa: S311
    assert {i["kind"] for i in items} == {"html", "emoji", "url", "mojibake"}
    assert all(i["value"] in text for i in items)
    assert next(i for i in items if i["kind"] == "mojibake")["value"] == "cafÃ©"
    cleaned = text
    for item in items:
        if item["kind"] != "emoji":
            cleaned = cleaned.replace(item["value"], "")
    result = score_noise({"d": items}, {"d": cleaned})
    assert result["emoji"]["removal_rate"] == 0.0
    assert result["url"]["removal_rate"] == 1.0


def test_label_maps_target_sct_types():
    sct_types = {"PERSON", "LOCATION", "ORGANISATION", "EMAIL", "PHONE", "IP", "URL", "DATE", "NUMBER"}
    for name, mapping in DATASET_LABELS.items():
        assert set(mapping.values()) <= sct_types, name


def test_summary_renders_without_text(tmp_path):
    rec = record("Anna", [span(0, 4, "GIVENNAME")])
    scores = score_pii([rec], {"d1": [span(0, 4, "PERSON")]}, {"d1": "<PERSON_0>"}, MAP)
    scores["time"] = {"docs_per_second": 1.0}
    (tmp_path / "x.json").write_text(
        json.dumps({"suite": "pii", "configs": {"default": scores}}), encoding="utf-8")
    (tmp_path / "y.json").write_text('{"suite": "pii", "error": "gated"}', encoding="utf-8")
    md = summary(tmp_path)
    assert "| x | default | 1 | 100.0 |" in md
    assert "- y: gated" in md
    assert "Anna" not in md


def test_wheel_excludes_benchmarks():
    """pyproject's package find include must pick sct and never benchmarks/ (setuptools fnmatch rule)."""
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]["packages"]["find"]
    top_level = [p.parent.name for p in ROOT.glob("*/__init__.py")]
    assert "benchmarks" in top_level

    def shipped(name):
        return (any(fnmatch.fnmatchcase(name, pattern) for pattern in config["include"])
                and not any(fnmatch.fnmatchcase(name, pattern) for pattern in config.get("exclude", ())))

    assert [name for name in top_level if shipped(name)] == ["sct"]


def test_adjacent_bio_person_entities_stay_separate():
    # Codex bench round 1 #3: two B-PER entities are two people, not one invented name.
    text = "Alice Bob"
    gold, _ = normalise_gold(text, [span(0, 5, "PER"), span(6, 9, "PER")], DATASET_LABELS["naamapadam"])
    assert gold == [span(0, 5, "PERSON"), span(6, 9, "PERSON")]
    rec = record(text, [span(0, 5, "PER"), span(6, 9, "PER")])
    preds = {"d1": [span(0, 5, "PERSON"), span(6, 9, "PERSON")]}
    result = score_pii([rec], preds, {"d1": "<PERSON_0> <PERSON_1>"}, DATASET_LABELS["naamapadam"])
    assert result["strict"]["overall"]["f1"] == 1.0


def test_leak_counts_unicode_folded_value():
    # Codex bench round 1 #4: SCT folds full-width text, so the name is still visible.
    assert leak("Ａｌｉｃｅ", "Hi Alice") == "full"
    assert leak("Ａｌｉｃｅ Ｂｅｒｇ", "Hi <PERSON_0> Berg") == "partial"
    assert leak("Ａｌｉｃｅ", "Hi <PERSON_0>") is None


def test_leak_by_language_excludes_dates_like_overall():
    # Codex bench round 1 #5: per-language rate uses the same denominator as overall.
    text = "Anna on 2020-01-01"
    rec = record(text, [span(0, 4, "GIVENNAME"), span(8, 18, "DATE")])
    result = score_pii([rec], {"d1": [span(0, 4, "PERSON")]}, {"d1": "<PERSON_0> on 2020-01-01"}, MAP)
    assert result["leak"]["overall"]["leak_rate"] == 0.0
    assert result["leak"]["by_language"]["en"] == result["leak"]["overall"]
    assert result["leak"]["by_label"]["DATE"]["full"] == 1


def test_leak_keeps_brahmic_vowel_signs_in_words():
    # \w alone splits these words at vowel signs into 1-2 char pieces, so no partial leak was ever seen.
    assert leak("राजेश कुमार", "मिलिए <PERSON_0> कुमार से") == "partial"
    assert leak("ரமேஷ் குமார்", "<PERSON_0> குமார்") == "partial"
    assert leak("राजेश कुमार", "मिलिए <PERSON_0> से") is None


def test_masked_recall_counts_any_type_overlap():
    text = "Call 0612345678 now"
    rec = record(text, [span(5, 15, "ID")])
    result = score_pii([rec], {"d1": [span(5, 15, "PHONE")]}, {"d1": "Call <PHONE_0> now"},
                       {"ID": "NUMBER", "TEL": "PHONE"})
    overall = result["partial"]["overall"]
    assert overall["recall"] == 0.0  # typed: PHONE is not NUMBER
    assert overall["gold_masked"] == 1 and overall["masked_recall"] == 1.0
    assert result["partial"]["by_label"]["NUMBER"]["masked_recall"] == 1.0
    assert result["partial"]["by_language"]["en"]["masked_recall"] == 1.0
