import json
import re

import pytest

import sct.report_cli as report_cli
from sct.utils.process_result import ProcessResult

ROWS = [
    {"text": "Mail anna@example.com about the quarterly budget review today",
     "city": "Utrecht", "age": "30-39", "secret": "do-not-report"},
    {"text": "Mail bob@example.org about the quarterly budget review today",
     "city": "Utrecht", "age": "30-39", "secret": "do-not-report"},
    {"text": "", "city": "Delft", "age": "50-59", "secret": "do-not-report"},
]


def _jsonl(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in ROWS) + "\n")
    return path


class _FakeCleaner:
    """Stands in for TextCleaner until ProcessResult.signals (K1) is merged."""

    def __init__(self, cfg):
        self.cfg = cfg

    def process_batch(self, texts):
        out = []
        for text in texts:
            result = type("R", (), {})()
            result.lm_text = re.sub(r"\S+@\S+", "<EMAIL>", text)
            result.signals = {
                "language": "ENGLISH" if text else None,
                "language_confidence": 0.99 if text else None,
                "pii_count": int("@" in text), "pii_share": 0.3 if "@" in text else 0.0,
                "pii_types": {"EMAIL": 1} if "@" in text else {},
                "warnings": [] if text else ["empty_input"],
                "models": [],
            }
            out.append(result)
        return out


def test_report_cli_with_fake_cleaner(tmp_path, monkeypatch):
    pytest.importorskip("datasketch")
    monkeypatch.setattr(report_cli, "TextCleaner", _FakeCleaner)
    out, card, scores = tmp_path / "r.json", tmp_path / "c.md", tmp_path / "s.jsonl"

    assert report_cli.main([
        str(_jsonl(tmp_path)), "--no-ner", "--minhash", "--quasi-identifiers", "city,age",
        "--k", "2", "--out", str(out), "--card", str(card), "--scores", str(scores),
    ]) == 0

    report = json.loads(out.read_text())
    assert report["summary"]["records"] == 3
    assert report["summary"]["pii_types"] == {"EMAIL": 2}
    assert report["minhash"]["num_perm"] == 128
    assert report["k_anonymity"]["quasi_identifiers"] == ["city", "age"]
    assert report["k_anonymity"]["at_risk_records"] == 1
    assert report["k_anonymity"]["class_size_counts"] == {"1": 1, "2": 1}
    text = out.read_text() + card.read_text() + scores.read_text()
    for leaked in ("do-not-report", "anna@", "Utrecht", "Delft"):
        assert leaked not in text
    rows = [json.loads(line) for line in scores.read_text().splitlines()]
    assert rows[0]["sct_minhash"] == rows[1]["sct_minhash"] is not None
    assert "# Data card" in card.read_text()


def test_report_cli_errors(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(report_cli, "TextCleaner", _FakeCleaner)
    src = _jsonl(tmp_path)
    out = str(tmp_path / "r.json")

    with pytest.raises(SystemExit):
        report_cli.main([str(src), "--quasi-identifiers", "zip", "--out", out])
    assert "missing" in capsys.readouterr().err
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"text": 3}) + "\n")
    with pytest.raises(SystemExit):
        report_cli.main([str(bad), "--out", out])
    assert "not text" in capsys.readouterr().err


def test_report_cli_reads_parquet(tmp_path, monkeypatch):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    monkeypatch.setattr(report_cli, "TextCleaner", _FakeCleaner)
    src = tmp_path / "in.parquet"
    pq.write_table(pa.Table.from_pylist(ROWS), src)
    out = tmp_path / "r.json"

    assert report_cli.main([str(src), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["summary"]["records"] == 3


@pytest.mark.skipif(not hasattr(ProcessResult, "signals"),
                    reason="needs ProcessResult.signals (K1)")
def test_report_cli_end_to_end_real_cleaner_ner_off(tmp_path):
    out = tmp_path / "r.json"

    assert report_cli.main([str(_jsonl(tmp_path)), "--no-ner", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["summary"]["records"] == 3
    assert report["summary"]["pii_types"] == {"EMAIL": 2}
    assert report["ner"] is False


# --- k-anonymity keeps values of different types in separate classes ---
@pytest.mark.parametrize("a,b", [([1], "[1]"), (True, 1)])
def test_report_cli_k_anonymity_distinct_types(tmp_path, a, b):
    src = tmp_path / "a.jsonl"
    src.write_text("".join(json.dumps({"text": t, "q": v}) + "\n" for t, v in [("x", a), ("y", b)]))
    out = tmp_path / "r.json"
    assert report_cli.main([str(src), "--no-ner", "--quasi-identifiers", "q", "--k", "2",
                            "--out", str(out)]) == 0
    k = json.loads(out.read_text())["k_anonymity"]
    assert k["minimum_k"] == 1 and k["at_risk_records"] == 2
