"""Codex 0.8.0 full review, round 4: ids 1, 2."""

import json

import pytest

import sct.report_cli as report_cli
from sct.presets import preset_config
from sct.risk import assess_reidentification_risk

FAST = preset_config("training", check_ner_process=False)


# --- 1: remove_columns with output="new_column" really drops the raw column ---

@pytest.mark.parametrize("iterable", [False, True])
def test_clean_dataset_remove_columns_drops_raw_text(iterable):
    datasets = pytest.importorskip("datasets")
    from sct.integrations.hf import clean_dataset

    ds = datasets.Dataset.from_dict({"text": ["mail alice@example.com"], "id": [7]})
    if iterable:
        ds = ds.to_iterable_dataset()
    out = clean_dataset(ds, output="new_column", config=FAST, remove_columns=["text"])
    rows = list(out)
    assert rows == [{"id": 7, "text_clean": "mail <EMAIL>"}]


# --- 2: k-anonymity keeps values of different types in separate classes ---

@pytest.mark.parametrize("a,b", [([1], "[1]"), (True, 1)])
def test_report_cli_k_anonymity_distinct_types(tmp_path, a, b):
    src = tmp_path / "a.jsonl"
    src.write_text("".join(json.dumps({"text": t, "q": v}) + "\n" for t, v in [("x", a), ("y", b)]))
    out = tmp_path / "r.json"
    assert report_cli.main([str(src), "--no-ner", "--quasi-identifiers", "q", "--k", "2",
                            "--out", str(out)]) == 0
    k = json.loads(out.read_text())["k_anonymity"]
    assert k["minimum_k"] == 1 and k["at_risk_records"] == 2


def test_risk_bool_and_int_are_distinct():
    risk = assess_reidentification_risk([{"q": True}, {"q": 1}], quasi_identifiers=("q",), k_threshold=2)
    assert risk.minimum_k == 1
