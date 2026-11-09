"""Release gates and sample identity must fail closed before an unattended comparison."""

import unittest
from pathlib import Path
import json
import tempfile
from unittest.mock import patch

from scripts.watch_release_comparison import REQUIRED_MERGE, compare_samples, main, release_candidate


class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.release = {"tag_name": "v0.8.3", "published_at": "2026-11-06T12:00:00Z", "draft": False,
                        "prerelease": False}
        self.comparison = {"status": "ahead", "commits": [{"sha": "a" * 40}], "base_commit": {"sha": "b" * 40}}

    def test_published_descendant_is_accepted(self):
        self.assertEqual(release_candidate(self.release, self.comparison, "a" * 40)["sha"], "a" * 40)

    def test_tag_commit_does_not_depend_on_paginated_commit_list(self):
        self.assertEqual(release_candidate(self.release, self.comparison, "c" * 40)["sha"], "c" * 40)

    def test_old_release_draft_and_prerelease_are_rejected(self):
        for change in ({"draft": True}, {"prerelease": True}, {"published_at": None}, {"tag_name": "v0.9.0"}):
            self.assertIsNone(release_candidate({**self.release, **change}, self.comparison, "a" * 40))
        self.assertIsNone(release_candidate(self.release, {**self.comparison, "status": "behind"}, "a" * 40))
        self.assertIsNone(release_candidate(self.release, {**self.comparison, "status": "diverged"}, "a" * 40))

    def test_release_exactly_at_merge_is_accepted(self):
        self.assertEqual(release_candidate(self.release, {**self.comparison, "status": "identical"}, "b" * 40)["sha"],
                         "b" * 40)

    def test_sample_mismatch_or_missing_hash_stops_comparison(self):
        left = {"tab": {"sample_sha256": "x"}}
        compare_samples(left, left)
        for right in ({}, {"tab": {}}, {"tab": {"sample_sha256": "y"}}):
            with self.assertRaises(ValueError):
                compare_samples(left, right)

    def test_poll_checks_ancestry_of_resolved_sha_not_mutable_tag(self):
        with tempfile.TemporaryDirectory() as folder, patch("scripts.watch_release_comparison.api") as api:
            api.side_effect = [self.release, {"sha": "c" * 40}, {**self.comparison, "status": "behind"}]
            main(Path(folder))
            self.assertTrue(api.call_args.args[0].endswith(f"{REQUIRED_MERGE}...{'c' * 40}"))

    def test_finished_or_failed_run_cannot_start_again(self):
        for phase in ("completed", "failed", "preparing", "running"):
            with tempfile.TemporaryDirectory() as folder, patch("scripts.watch_release_comparison.api") as api:
                root = Path(folder)
                (root / "state.json").write_text(json.dumps({"phase": phase}))
                main(root)
                api.assert_not_called()
                if phase in {"preparing", "running"}:
                    self.assertEqual(json.loads((root / "state.json").read_text())["phase"], "failed")

    def test_valid_published_release_launches_the_comparison(self):
        with tempfile.TemporaryDirectory() as folder, patch("scripts.watch_release_comparison.api") as api:
            with patch("scripts.watch_release_comparison.execute") as execute:
                api.side_effect = [self.release, {"sha": "c" * 40}, self.comparison]
                main(Path(folder))
                execute.assert_called_once()
                self.assertEqual(execute.call_args.args[1]["sha"], "c" * 40)

    def test_setup_failure_is_recorded_and_cannot_retry_forever(self):
        with tempfile.TemporaryDirectory() as folder, patch("scripts.watch_release_comparison.api") as api:
            with patch("scripts.watch_release_comparison.prepare", side_effect=ValueError("invalid package version")):
                root = Path(folder)
                api.side_effect = [self.release, {"sha": "c" * 40}, self.comparison]
                with self.assertRaises(ValueError):
                    main(root)
                state = json.loads((root / "state.json").read_text())
                self.assertEqual(state["phase"], "failed")
                self.assertIn("invalid package version", state["step"])
