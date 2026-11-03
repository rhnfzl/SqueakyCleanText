"""Round-2 dataset loaders (PROTOCOL.md section 1): registry, span building, own splits, untyped and
must-survive scoring. Hand-written fixtures only, no dataset text."""

import json

import pytest

from benchmarks.compare import entries, runner, scorer, split
from benchmarks.datasets import MODULES, ROUND2, get, sha256_checked, valid_spans
from benchmarks.datasets.dialogpii import dialog_record
from benchmarks.datasets.greek_deid import note_record
from benchmarks.datasets.kdpii import sentence_record
from benchmarks.datasets.meddeid import doc_record
from benchmarks.datasets.nus_lt import gold_label
from benchmarks.datasets.pl_pii import entity_spans
from benchmarks.datasets.redmadrobot import token_spans
from benchmarks.datasets.tonic import row_record
from benchmarks.labels import DATASET_LABELS, normalise_gold, unknown_gold_labels


def span(start, end, label):
    return {"start": start, "end": end, "label": label}


def test_round2_registered_pinned_grouped_and_clean_for_sct():
    grouped = {d for members in scorer.GROUPS.values() for d in members}
    for name in ROUND2:
        module = get(name)
        assert name in MODULES and name in grouped
        assert module.PUBLISH == "scores-only" and module.REVISION  # SHORTLIST section 3: no per-example output
        if module.HF_ID.count("/") == 1:  # HF repos pin a full commit sha
            assert len(module.REVISION) == 40, name
        assert name in DATASET_LABELS or name == "carmen"  # carmen's map waits for its annotation.conf
        assert entries.status("sct-default", name) == "clean"
    assert {n for n in ROUND2 if getattr(get(n), "DELETE_PREDICTIONS", False)} == {"nus_lt", "carmen"}
    assert not set(scorer.GROUPS["r2-untyped"]) & {d for g, m in scorer.GROUPS.items() if g != "r2-untyped" for d in m}


def test_carmen_stub_says_it_needs_physionet_access():
    with pytest.raises(RuntimeError, match="PhysioNet access"):
        get("carmen").load(None, 13)


def test_split_keeps_a_datasets_own_part():
    records = [{"id": str(i), "lang": "tr", "text": "x", "spans": [], "part": "dev" if i < 3 else "test"}
               for i in range(10)]
    assert [r["id"] for r in split.split("neonredact", records, "dev")] == ["0", "1", "2"]
    assert len(split.split("neonredact", records, "test")) == 7
    hashed = [{"id": str(i), "lang": "en", "text": "x", "spans": []} for i in range(500)]
    assert 0 < len(split.split("newmean", hashed, "dev")) < 150  # no part: seed-13 hash split as before


def test_loaders_with_own_split_tag_parts(monkeypatch, tmp_path):
    rows = {"dev": {"id": "1", "kategori": "log", "text": "Ali Veli", "entities": [span(0, 8, "KISI")]},
            "test": {"id": "2", "kategori": "log", "text": "yok", "entities": []}}
    for part, row in rows.items():
        (tmp_path / f"{part}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    module = get("neonredact")
    monkeypatch.setattr(module, "hf_file", lambda repo, path, rev: tmp_path / path)
    monkeypatch.setattr(module, "SHA256", {})  # the pins hold for the real files only
    records = module.load(None, 13)
    assert [(r["id"], r["part"]) for r in records] == [("dev:1", "dev"), ("test:2", "test")]
    assert records[0]["spans"] == [span(0, 8, "KISI")]


def test_dialogpii_turns_join_with_speaker_and_shift_spans():
    turns = [{"speaker": "A", "text": "Hi Ann", "annotations": [{"type": "PERSON", "start": 3, "end": 6}]},
             {"speaker": "BB", "text": "Bo here", "annotations": [{"type": "PERSON", "start": 0, "end": 2}]}]
    text, spans = dialog_record(turns)
    assert text == "A: Hi Ann\nBB: Bo here"
    assert [text[s["start"]:s["end"]] for s in spans] == ["Ann", "Bo"]


def test_kdpii_sentence_record_keeps_part_and_dialog_group():
    row = {"sent_idx": "Data3.12_1_0", "sentence": "x 0101", "PII_set": [{"form": "0101", "label": "QT_MOBILE",
                                                                          "begin": 2, "end": 6}]}
    record = sentence_record(row, "dev")
    assert record["part"] == "dev" and record["group"] == "Data3.12" and record["spans"] == [span(2, 6, "QT_MOBILE")]


def test_nus_lt_misc_split_by_shape_of_gold_value():
    text = "a@b.lt www.x.lt Foo"
    assert [gold_label(text, s) for s in (span(0, 6, "MISC"), span(7, 15, "MISC"), span(16, 19, "MISC"),
                                          span(16, 19, "PER"))] == ["MISC:email", "MISC:url", "MISC", "PER"]


def test_redmadrobot_tokens_align_case_insensitively_and_names_merge():
    text = "Ann Lee, www.x.ru"
    spans = token_spans(text, ["Ann", "Lee", ",", "WWW", ".", "x", ".", "ru"],
                        ["B-FIRST_NAME", "B-LAST_NAME", "O", "B-URL", "I-URL", "I-URL", "I-URL", "I-URL"])
    assert spans == [span(0, 3, "FIRST_NAME"), span(4, 7, "LAST_NAME"), span(9, 17, "URL")]
    gold, _ = normalise_gold(text, spans, DATASET_LABELS["redmadrobot"])
    assert gold == [span(0, 7, "PERSON"), span(9, 17, "URL")]
    assert token_spans("ab", ["zz", "b"], ["B-CITY", "I-CITY"]) == [span(1, 2, "CITY")]  # unfound token skipped


def test_meddeid_birthdates_ages_contacts_and_final_eval_only():
    row = {"document_id": "d1", "text": "43 y 1/2/1980 a@b.be 0470",
           "metadata_json": json.dumps({"lang": "nl-BE"}),
           "spans": [{"begin": 0, "end": 4, "text": "43 y", "category": "Age_Birthdate", "subannotations": []},
                     {"begin": 5, "end": 13, "text": "1/2/1980", "category": "Age_Birthdate",
                      "subannotations": [{"category": "day"}, {"category": "year"}]},
                     {"begin": 14, "end": 20, "text": "a@b.be", "category": "Contactdetails"},
                     {"begin": 21, "end": 25, "text": "0470", "category": "Contactdetails"}]}
    record = doc_record(row, "nl")
    assert record["lang"] == "nl-be" and record["part"] == "test" and record["id"] == "nl-be:d1"
    assert [s["label"] for s in record["spans"]] == ["Age_Birthdate:age", "Age_Birthdate:date",
                                                     "Contactdetails:email", "Contactdetails:phone"]
    gold, out = normalise_gold(record["text"], record["spans"], DATASET_LABELS["meddeid"])
    assert [g["label"] for g in gold] == ["DATE", "EMAIL", "PHONE"] and out[0]["label"] == "Age_Birthdate:age"


def test_pl_pii_keep_entities_are_out_of_scope():
    spans = entity_spans("Jan Nowak", [{"start": 0, "end": 3, "label": "PERSON", "protection": "protect"},
                                       {"start": 4, "end": 9, "label": "PERSON", "protection": "keep"}])
    assert [s["label"] for s in spans] == ["PERSON", "PERSON:keep"]


def test_tonic_groups_pages_of_a_document_and_slack_threads():
    page = {"meta": {"filename": "a.docx", "page": 2}, "text": "Ann", "ground_truth_spans": [span(0, 3, "NAME_GIVEN")]}
    msg = {"meta": {"row_id": "1.2", "thread_ts": "1.0"}, "text": "hi", "ground_truth_spans": []}
    assert row_record("w", "doc", 5, page)["group"] == "w:doc:a.docx"
    assert row_record("w", "msg", 0, msg)["group"] == "w:msg:1.0"


def test_greek_survive_probes_and_untyped_gold_scoring():
    row = {"id": "n1", "text": "Ann took aspirin 5 mg",
           "phi": [{"expect": "survive", "value": "aspirin"}, {"expect": "survive", "value": "5 mg"},
                   {"expect": "redact", "value": "Ann"}]}
    record = note_record(row, [{"start": 0, "end": 3, "fine": "greek_patient_name"}])
    assert record["survive"] == ["aspirin", "5 mg"] and record["spans"] == [span(0, 3, "greek_patient_name")]
    outputs = {"n1": {"id": "n1", "spans": [span(0, 3, "PERSON"), span(17, 18, "NUMBER")]}}
    views = scorer.views_for(("PERSON",), ("PERSON",), ("PERSON", "NUMBER"))
    result, _ = scorer.score_run([record], outputs, {"PERSON": "PERSON", "NUMBER": "NUMBER"},
                                 DATASET_LABELS["greek_deid"], views)
    assert result["survive"] == {"probes": 2, "kept": 1, "survive_rate": 0.5}  # "5 mg" lost its digit

    untyped = {"id": "r1", "lang": "en", "text": "Ann Lee, form 12",
               "spans": [span(0, 3, "mandatory"), span(3, 4, "contextual"), span(4, 7, "mandatory")]}
    outputs = {"r1": {"id": "r1", "spans": [span(0, 7, "PERSON"), span(14, 16, "NUMBER")]}}
    result, _ = scorer.score_run([untyped], outputs, {"PERSON": "PERSON", "NUMBER": "NUMBER"},
                                 DATASET_LABELS["redactionbench"], views)
    micro = result["views"]["supported"]["compare"]["micro"]  # untyped gold is in every view
    assert micro["gold"] == 2 and micro["masked_recall"] == 1.0
    assert micro["strict_f1"] is None and micro["partial_f1"] is None  # no type to match
    assert result["survive"] is None


def test_surface_form_and_sha256_pins_fail_loudly(tmp_path):
    spans = [{"start": 2, "end": 6, "label": "QT_MOBILE", "form": "0101"}, {"start": 0, "end": 1, "label": "X"}]
    assert valid_spans("x 0101", spans) == [span(2, 6, "QT_MOBILE"), span(0, 1, "X")]  # form is checked, then dropped
    with pytest.raises(ValueError, match="1 of 1 gold spans"):  # shifted by one: a wrong offset convention
        valid_spans("xx 0101", [{"start": 2, "end": 6, "label": "QT_MOBILE", "form": "0101"}])
    path = tmp_path / "f"
    path.write_bytes(b"abc")
    assert sha256_checked(path, None) == path
    assert sha256_checked(path, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") == path
    with pytest.raises(ValueError, match="sha256"):
        sha256_checked(path, "0" * 64)


def test_greek_gold_ids_must_match_the_corpus(monkeypatch, tmp_path):
    module = get("greek_deid")
    (tmp_path / "data").mkdir()
    (tmp_path / "annotations").mkdir()
    (tmp_path / module.CORPUS).write_text(json.dumps({"id": "n1", "text": "Ann"}) + "\n", encoding="utf-8")
    (tmp_path / module.GOLD).write_text(json.dumps({"notes": {"n2": []}}), encoding="utf-8")
    monkeypatch.setattr(module, "hf_file", lambda repo, path, rev: tmp_path / path)
    monkeypatch.setattr(module, "SHA256", {})
    with pytest.raises(ValueError, match="gold ids"):
        module.load(None, 13)


def test_survival_counts_occurrences_of_short_probes():
    record = {"id": "n1", "text": "5 mg then 5 mg", "survive": ["5 mg"]}
    assert scorer.survival([record], {"n1": []})["kept"] == 1
    assert scorer.survival([record], {"n1": [span(0, 1, "NUMBER")]})["kept"] == 0  # one copy masked: lost


def test_unknown_gold_labels_are_counted_never_silently_out_of_scope():
    records = [{"spans": [span(0, 1, "PS_NAME"), span(0, 1, "PS_ID"), span(0, 1, "DT_WEDDING")]}]
    assert unknown_gold_labels("kdpii", records) == {"DT_WEDDING": 1}
    assert unknown_gold_labels("pl_pii", [{"spans": [span(0, 1, "PERSON:keep")]}]) == {}
    assert unknown_gold_labels("gretel_en", records) == {}  # round 1: no declared list, no check
    assert set(DATASET_LABELS) >= {d for d in ROUND2 if d != "carmen"}


def test_runner_deletes_prediction_caches_and_refuses_mixed_rounds(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="mixes round-1 and round-2"):
        runner.shared_view(["gretel_en", "kdpii"], {"sct-default": {}})
    records = [{"id": "a", "lang": "lt", "text": "mail a@b.lt now", "spans": [span(5, 11, "MISC:email")]}]
    monkeypatch.setattr(runner.split, "load", lambda dataset, part, n, seed: records)
    cache, out = tmp_path / "cache", tmp_path / "out"
    runner.main(["--entries", "dummy", "--datasets", "nus_lt", "--part", "dev", "--out", str(out), "--cache", str(cache),
                 "--doc-timeout", "30", "--start-timeout", "30", "--skip-offset-check"])
    result = json.loads((out / "dummy" / "nus_lt-dev.json").read_text())
    assert result["process"]["predictions_deleted"] is True and "failed_ids" not in result
    assert not list((cache / "dummy").glob("nus_lt-dev*")) and not (out / "predictions").exists()
    records[0]["spans"].append(span(0, 4, "NEW_LABEL"))
    with pytest.raises(SystemExit, match="NEW_LABEL"):
        runner.main(["--entries", "dummy", "--datasets", "nus_lt", "--part", "dev", "--out", str(out),
                     "--cache", str(cache), "--skip-offset-check"])
