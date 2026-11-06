"""Chart differences must retain the scorer's direction and every non-SCT entry."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_pii_comparison import extract

SOURCE = Path(__file__).resolve().parents[1] / "benchmarks/results/compare-2026-11-04/default/summary.json"


class PIIChartTests(unittest.TestCase):
    def test_all_entries_and_shared_recall_tradeoff(self):
        data = extract(SOURCE)
        self.assertEqual(len(data["rows"]), 14)
        self.assertEqual(data["sct_version"], "0.8.2")
        worse = [r["entry"] for r in data["rows"] if r["shared"]["metrics"]["masked_recall"]["ci95"][1] < 0]
        self.assertEqual(worse, ["bardsai-v3", "openmed", "opf", "pii-tracer"])
        full = {r["entry"]: r["full"]["metrics"]["masked_recall"] for r in data["rows"]}
        self.assertAlmostEqual(full["pii-tracer"]["difference"], 0.0504)
        self.assertLess(full["gliner2-fastino"]["ci95"][0], 0)
        self.assertGreater(full["gliner2-fastino"]["ci95"][1], 0)

    def test_reversed_direction_or_missing_entry_stops_export(self):
        data = json.loads(SOURCE.read_text())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "summary.json"
            data["paired_note"] = "a = entry, b = sct-default"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "direction"):
                extract(path)
            data["paired_note"] = "a = sct-default, b = entry"
            del data["pairwise"]["pii-tracer"]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "14"):
                extract(path)
