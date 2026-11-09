"""Social cards keep scope, uncertainty and the direction of each measure visible."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_pii_social_cards import build
from scripts.plot_pii_comparison import extract

SOURCE = Path(__file__).resolve().parents[1] / "benchmarks/results/compare-2026-11-04/default/summary.json"


class SocialCardTests(unittest.TestCase):
    def test_cards_preserve_claims_and_large_labels(self):
        models = {}
        with (
            tempfile.TemporaryDirectory() as folder,
            patch(
                "scripts.build_pii_social_cards.Card.write",
                autospec=True,
                side_effect=lambda card, path: models.update({path.stem: card.model}),
            ),
        ):
            output = Path(folder)
            metadata = build(extract(SOURCE), output)
            self.assertEqual(metadata["full_counts"], [13, 1, 0])
            self.assertEqual(len(metadata["alt_text"]), 6)
            self.assertTrue(all(len(text) <= 1000 for text in metadata["alt_text"].values()))
            self.assertEqual(len(models), 6)
            labels = {}
            for name, root in models.items():
                labels[name] = " ".join(cell.get("value", "") for cell in root.iter("mxCell"))
                ids = [cell.get("id") for cell in root.iter("mxCell")]
                self.assertEqual(len(ids), len(set(ids)))
                for cell in root.iter("mxCell"):
                    for part in cell.get("style", "").split(";"):
                        if part.startswith("fontSize="):
                            self.assertGreaterEqual(int(part.split("=")[1]), 36)
            self.assertIn("Percentage points above SCT default", labels["03-shared-recall"])
            self.assertIn("+13.5 points", labels["03-shared-recall"])
            self.assertIn("Unsupported types count as misses", labels["04-full-recall"])
            self.assertIn("78.1%", labels["05-precision"])
            self.assertIn("75.0%", labels["05-precision"])
            self.assertIn("58.0%", labels["06-exposed-text"])
            self.assertIn("50.2%", labels["06-exposed-text"])
            self.assertIn("lower is better", labels["06-exposed-text"])
            self.assertIn("5.7 to 10.1 points", labels["06-exposed-text"])

    def test_wrong_version_or_incomplete_comparison_stops(self):
        data = extract(SOURCE)
        with tempfile.TemporaryDirectory() as folder:
            for changed in ({**data, "sct_version": "0.8.3"}, {**data, "rows": data["rows"][:-1]}):
                with self.assertRaises(ValueError):
                    build(changed, Path(folder))
