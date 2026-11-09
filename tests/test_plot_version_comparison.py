"""Keep charts tied to the recorded versions, settings and samples."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.plot_version_comparison import paired, summarize


class PlotComparisonTests(unittest.TestCase):
    def test_recorded_summary_matches_published_values(self):
        root = Path(__file__).resolve().parents[1] / "benchmarks/results/compare-0.7.1-0.8.3"
        summary = summarize(root)
        self.assertEqual(sum(row["documents"] for row in summary["quality"]), 16372)
        self.assertEqual(summary["speed_docs_per_second"]["0.8.3"]["default"], [2.669, 2.677, 2.662])
        self.assertEqual(summary["limits"]["0.8.3"]["tab_direct_full_coverage"], 0.4022)

    def test_mismatched_samples_settings_and_versions_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = {"sample_sha256": "abc", "documents": 1, "languages": ["en"], "seed": 13,
                    "n_per_language": None, "suite": "pii", "dataset": "example",
                    "run": {"scoring_overrides": {}, "threads": 16, "workers": 2, "configs": ["default"]}}
            left = {**base, "run": {**base["run"], "sct_version": "0.7.1"}}
            right = {**base, "run": {**base["run"], "sct_version": "0.8.3"}}
            (root / "0.7.1.json").write_text(json.dumps(left))
            for changed in ({**right, "sample_sha256": "different"},
                            {**right, "sample_sha256": ""},
                            {**right, "run": {**right["run"], "threads": 32}},
                            {**right, "run": {**right["run"], "sct_version": "0.8.2"}}):
                (root / "0.8.3.json").write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    paired(root, "{version}.json")
            (root / "0.8.3.json").write_text(json.dumps(right))
            self.assertEqual(len(paired(root, "{version}.json")), 2)
