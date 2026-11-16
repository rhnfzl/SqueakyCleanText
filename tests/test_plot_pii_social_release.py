"""Published release charts must retain their measured source and version."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts.plot_pii_social_release import extract

BUNDLE = Path(__file__).resolve().parents[1] / "benchmarks/results/compare-2026-11-12-v010/default"


def test_published_evidence_hashes_and_recall_directions():
    checked = json.loads((BUNDLE / "validation.json").read_text())
    assert checked["passed"] is True
    for name, key in [("summary.json", "summary_sha256"), ("native-speed.json", "native_speed_sha256")]:
        assert hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == checked[key]
    assert len(checked["result_files"]) == 180
    for record in checked["result_files"]:
        path = BUNDLE / record["entry"] / f"{record['dataset']}-test.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    data = extract(BUNDLE)
    assert data["sct_version"] == "0.10.0" and len(data["rows"]) == 14
    assert sum(r["shared"]["metrics"]["masked_recall"]["ci95"][1] < 0 for r in data["rows"]) == 4
    assert sum(r["full"]["metrics"]["masked_recall"]["ci95"][0] > 0 for r in data["rows"]) == 13


@pytest.mark.parametrize("change", ["version", "summary"])
def test_wrong_version_or_changed_summary_cannot_be_labelled_as_release(tmp_path, change):
    for name in ["summary.json", "validation.json"]:
        shutil.copyfile(BUNDLE / name, tmp_path / name)
    if change == "version":
        path = tmp_path / "validation.json"
        data = json.loads(path.read_text())
        data["sct_version"] = "0.8.2"
        path.write_text(json.dumps(data))
    else:
        with (tmp_path / "summary.json").open("ab") as output:
            output.write(b"\n")
    with pytest.raises(AssertionError):
        extract(tmp_path)
