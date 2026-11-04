"""Dataset loaders on small local files: ids, languages, spans. No download; pyarrow is faked where it is read."""

import json
import sys
import zipfile
from types import SimpleNamespace

import pytest

from benchmarks.datasets import (
    argilla_pii,
    bayanat,
    fineweb2,
    gretel,
    iahlt,
    multiconer,
    naamapadam,
    nemotron,
    openpii,
    redact,
    stockmark,
    tab,
    thai_pii,
    thainer,
    wili,
)


def pinned(module, monkeypatch, path, attr="hf_file"):
    """Serve *path* for any file of *module* and check the loader asks at its pinned revision."""
    def fake(repo, name, revision):
        assert (repo, revision) == (module.HF_ID, module.REVISION)
        return path(name) if callable(path) else path

    monkeypatch.setattr(module, attr, fake)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def write_json(path, rows):
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


class FakeTable:
    def __init__(self, rows):
        self.rows, self.num_rows = rows, len(rows)

    def to_pylist(self):
        return self.rows

    def column(self, name):
        return SimpleNamespace(to_pylist=lambda: [row[name] for row in self.rows])


@pytest.fixture
def parquet(monkeypatch):
    """A fake ``pyarrow.parquet`` that reads back the rows and names set on it."""
    state = SimpleNamespace(rows=[], metadata=None, columns=[])

    def read_table(path, columns=None):
        state.columns.append(columns)
        return FakeTable(state.rows(path) if callable(state.rows) else state.rows)

    pq = SimpleNamespace(read_table=read_table,
                         read_schema=lambda path: SimpleNamespace(metadata={b"huggingface": json.dumps(state.metadata)}))
    monkeypatch.setitem(sys.modules, "pyarrow", SimpleNamespace(parquet=pq))
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", pq)
    return state


def test_nemotron_id_is_uid_and_locale_and_bad_spans_are_dropped(monkeypatch, parquet):
    pinned(nemotron, monkeypatch, "file")
    parquet.rows = [{"uid": "u1", "locale": "us", "document_type": "memo", "text": "Ann here",
                     "spans": "[{'start': 0, 'end': 3, 'label': 'first_name', 'text': 'x'},"
                              " {'start': 4, 'end': 99, 'label': 'city'}]"}]
    assert nemotron.load(None, 1) == [{"id": "u1:us", "lang": "en", "text": "Ann here", "group": "memo",
                                       "spans": [{"start": 0, "end": 3, "label": "first_name"}]}]


def test_wili_labels_come_from_the_schema_and_the_row_index_is_the_id(monkeypatch, parquet):
    pinned(wili, monkeypatch, "file")
    parquet.metadata = {"info": {"features": {"label": {"names": ["nld", "eng"]}}}}
    parquet.rows = [{"sentence": "hallo", "label": 0}, {"sentence": "hello", "label": 1}]
    assert wili.load(None, 1) == [{"id": "0", "lang": "nld", "text": "hallo", "spans": []},
                                  {"id": "1", "lang": "eng", "text": "hello", "spans": []}]


def test_argilla_reads_a_missing_suggestion_as_no_spans(monkeypatch, parquet):
    pinned(argilla_pii, monkeypatch, "file")
    parquet.rows = [{"id": "a", "language": "en", "source-text": "Ann", "pii.suggestion": None},
                    {"id": "b", "language": "en", "source-text": "Bob",
                     "pii.suggestion": [{"start": 0, "end": 3, "label": "FIRSTNAME", "score": 1.0}]}]
    assert [r["spans"] for r in argilla_pii.load(None, 1)] == [[], [{"start": 0, "end": 3, "label": "FIRSTNAME"}]]


def test_gretel_maps_the_language_column_including_its_france_value(monkeypatch, parquet):
    pinned(gretel, monkeypatch, "file")
    parquet.rows = [{"index": 7, "language": "France", "document_type": "bill", "generated_text": "Paris",
                     "pii_spans": json.dumps([{"start": 0, "end": 5, "label": "city"}])}]
    assert gretel.load(None, 1) == [{"id": "7", "lang": "fr", "text": "Paris", "group": "bill",
                                     "spans": [{"start": 0, "end": 5, "label": "city"}]}]


def test_thainer_joins_words_with_no_separator_and_decodes_tag_ids(monkeypatch, parquet):
    pinned(thainer, monkeypatch, "file")
    parquet.metadata = {"info": {"features": {"ner": {"feature": {"names": ["O", "B-PER", "I-PER"]}}}}}
    parquet.rows = [{"words": ["คุณ", "สม", "ชาย"], "ner": [0, 1, 2]}]
    assert thainer.load(None, 1) == [{"id": "0", "lang": "th", "text": "คุณสมชาย",
                                      "spans": [{"start": 3, "end": 8, "label": "PER"}]}]


def test_fineweb2_samples_per_directory_from_the_seed(monkeypatch, parquet):
    pinned(fineweb2, monkeypatch, lambda name: name)
    parquet.rows = lambda path: [{"id": f"{path.split('/')[1]}-{i}", "text": f"t{i}"} for i in range(5)]
    picked = fineweb2.load(2, 13, dirs=("nld_Latn", "rus_Cyrl"))
    assert [r["lang"] for r in picked] == ["nld_Latn"] * 2 + ["rus_Cyrl"] * 2
    assert picked == fineweb2.load(2, 13, dirs=("nld_Latn", "rus_Cyrl"))
    assert all(r["id"].startswith(r["lang"]) and r["spans"] == [] for r in picked)
    assert len(fineweb2.load(None, 13, dirs=("nld_Latn",))) == 5
    assert len(fineweb2.load(99, 13, dirs=("nld_Latn",))) == 5
    assert parquet.columns[0] == ["id", "text"]


def test_openpii_mask_may_be_text_or_a_list(tmp_path, monkeypatch):
    span = {"start": 0, "end": 3, "label": "NAME", "value": "x"}
    pinned(openpii, monkeypatch, write_jsonl(tmp_path / "v.jsonl", [
        {"uid": 1, "language": "de", "source_text": "Ann", "privacy_mask": str([span])},
        {"uid": 2, "language": "de", "source_text": "Bob", "privacy_mask": [span, {**span, "end": 9}]},
    ]))
    wanted = [{"start": 0, "end": 3, "label": "NAME"}]
    assert openpii.load(None, 1) == [{"id": "1", "lang": "de", "text": "Ann", "spans": wanted},
                                     {"id": "2", "lang": "de", "text": "Bob", "spans": wanted}]


def test_naamapadam_reads_each_language_zip(tmp_path, monkeypatch):
    def archive(name):
        lang = name.split("/")[1].split("_")[0]
        path = tmp_path / f"{lang}.zip"
        with zipfile.ZipFile(path, "w") as out:
            out.writestr(f"{lang}_test.json", json.dumps({"words": ["a", "b"], "ner": ["B-PER", "O"]}) + "\n")
        return path

    pinned(naamapadam, monkeypatch, archive)
    records = naamapadam.load(None, 1)
    assert [r["id"] for r in records] == ["hi:0", "bn:0", "ta:0"]
    assert all(r["text"] == "a b" and r["spans"] == [{"start": 0, "end": 1, "label": "PER"}] for r in records)


def test_multiconer_parses_conll_and_joins_chinese_with_no_space(tmp_path, monkeypatch):
    conll = "# id u1\tdomain=x\nAnn _ _ B-PER\nruns _ _ O\n\n# id u2\nBob _ _ I-PER"

    def path(name):
        out = tmp_path / name.replace("/", "_")
        out.write_text(conll, encoding="utf-8")
        return out

    pinned(multiconer, monkeypatch, path)
    records = {r["lang"]: [] for r in multiconer.load(None, 1)}
    for record in multiconer.load(None, 1):
        records[record["lang"]].append(record)
    assert [r["id"] for r in records["hi"]] == ["u1", "u2"]  # the last sentence has no blank line after it
    assert records["hi"][0]["text"] == "Ann runs" and records["zh"][0]["text"] == "Annruns"
    assert records["zh"][0]["spans"] == [{"start": 0, "end": 3, "label": "PER"}]


def test_thai_pii_keeps_valid_entities(tmp_path, monkeypatch):
    pinned(thai_pii, monkeypatch, write_json(tmp_path / "t.json", [
        {"id": "x", "text": "abc", "entities": [{"start": 0, "end": 2, "label": "PHONE"},
                                                 {"start": 2, "end": 2, "label": "EMPTY"}]}]))
    assert thai_pii.load(None, 1) == [{"id": "x", "lang": "th", "text": "abc",
                                       "spans": [{"start": 0, "end": 2, "label": "PHONE"}]}]


def test_redact_drops_spans_that_differ_from_their_string_and_prefixes_the_row(tmp_path, monkeypatch):
    entities = [{"start": 0, "end": 3, "entity_type": "NAME", "entity_string": "Ann"},
                {"start": 4, "end": 7, "entity_type": "NAME", "entity_string": "Zed"}]
    pinned(redact, monkeypatch, write_json(tmp_path / "r.json", [
        {"record_id": "same", "text": "Ann Bob", "entities": str(entities), "axes": str({"language": "DE"})},
        {"record_id": "same", "text": "Ann Bob", "entities": entities, "axes": {"language": "DE"}},
    ]))
    records = redact.load(None, 1)
    assert [r["id"] for r in records] == ["0:same", "1:same"]
    assert all(r["lang"] == "de" and r["group"] == "same" for r in records)
    assert all(r["spans"] == [{"start": 0, "end": 3, "label": "NAME"}] for r in records)


def test_tab_uses_the_quality_checked_annotator_and_caps_by_doc_id(tmp_path, monkeypatch):
    def mention(start, end, identifier="DIRECT"):
        return {"start_offset": start, "end_offset": end, "entity_type": "PERSON", "identifier_type": identifier}

    docs = [{"doc_id": "b", "text": "Ann Bob", "quality_checked": ["ann2"],
             "annotations": {"ann1": {"entity_mentions": [mention(0, 3)]},
                             "ann2": {"entity_mentions": [mention(4, 7, "QUASI"), mention(5, 99)]}}},
            {"doc_id": "a", "text": "Cy", "quality_checked": None,
             "annotations": {"z": {"entity_mentions": []}, "y": {"entity_mentions": [mention(0, 2)]}}}]
    def fake_url_file(url, name):
        assert tab.REVISION in url and tab.REVISION in name
        return write_json(tmp_path / "t.json", docs)

    monkeypatch.setattr(tab, "url_file", fake_url_file)
    first, second = tab.load(None, 1)
    assert (first["id"], first["spans"]) == ("a", [{"start": 0, "end": 2, "label": "PERSON"}])  # sorted annotators
    assert second["spans"] == [{"start": 4, "end": 7, "label": "PERSON"}]
    assert second["annotators"]["ann2"] == [{"start": 4, "end": 7, "label": "PERSON", "identifier": "QUASI"}]
    assert [r["id"] for r in tab.load(1, 1)] == ["a"]


def test_iahlt_reads_label_triples(tmp_path, monkeypatch):
    pinned(iahlt, monkeypatch, write_jsonl(tmp_path / "i.jsonl", [
        {"id": 3, "text": "abcd", "label": [[0, 2, "PER"], [0, 4, "ORG"], [3, 9, "LOC"]]}]))
    assert iahlt.load(None, 1) == [{"id": "3", "lang": "ar", "text": "abcd",
                                    "spans": [{"start": 0, "end": 2, "label": "PER"},
                                              {"start": 0, "end": 4, "label": "ORG"}]}]  # nested stay


def test_stockmark_id_is_row_and_curid(tmp_path, monkeypatch):
    pinned(stockmark, monkeypatch, write_json(tmp_path / "s.json", [
        {"curid": "9", "text": "東京", "entities": [{"span": [0, 2], "type": "地名"}]}]))
    assert stockmark.load(None, 1) == [{"id": "0:9", "lang": "ja", "text": "東京",
                                        "spans": [{"start": 0, "end": 2, "label": "地名"}]}]


def test_bayanat_language_carries_the_variety(tmp_path, monkeypatch):
    pinned(bayanat, monkeypatch, write_jsonl(tmp_path / "b.jsonl", [
        {"id": "r", "variety": "gulf", "text": "abc", "spans": [{"start": 1, "end": 3, "entity_type": "PHONE"}]}]))
    assert bayanat.load(None, 1) == [{"id": "r", "lang": "ar-gulf", "text": "abc",
                                      "spans": [{"start": 1, "end": 3, "label": "PHONE"}]}]


def test_loaders_honour_the_per_language_sample(tmp_path, monkeypatch):
    rows = [{"id": str(i), "variety": "gulf", "text": "abc", "spans": []} for i in range(20)]
    pinned(bayanat, monkeypatch, write_jsonl(tmp_path / "b.jsonl", rows))
    sample = bayanat.load(3, 13)
    assert len(sample) == 3 and sample == bayanat.load(3, 13)
