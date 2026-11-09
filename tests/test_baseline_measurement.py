"""Measurements must count runtime calls and identify the measured source."""

import subprocess
from types import SimpleNamespace

from evaluation.improvements import measure_baseline


class Session:
    def __init__(self):
        self.invocations = 0

    def run(self, value):
        self.invocations += 1
        return value * 2


def test_shared_pipeline_counts_underlying_invocations_once():
    session = Session()
    shared = SimpleNamespace(_session=session)
    other = SimpleNamespace(_session=session)
    calls = measure_baseline.instrument_sessions([shared, shared, other])
    assert shared._session.run(3) == 6
    assert other._session.run(4) == 8
    assert session.invocations == 2
    assert calls[0] == session.invocations


def test_provenance_reports_actual_revision_and_dirty_source(tmp_path):
    subprocess.run(["/usr/bin/git", "init", str(tmp_path)], check=True, capture_output=True)  # noqa: S603 - fixed Git commands, resolved source path or pytest-owned fixture path; no shell.
    source = tmp_path / "sct"
    source.mkdir()
    module = source / "sample.py"
    module.write_text("VALUE = 1\n")
    subprocess.run(["/usr/bin/git", "-C", str(tmp_path), "add", "sct"], check=True)  # noqa: S603 - fixed Git commands, resolved source path or pytest-owned fixture path; no shell.
    subprocess.run(  # noqa: S603 - fixed Git commands, resolved source path or pytest-owned fixture path; no shell.
        [
            "/usr/bin/git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    identity = measure_baseline.source_identity(tmp_path)
    actual = subprocess.check_output(["/usr/bin/git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()  # noqa: S603 - fixed Git commands, resolved source path or pytest-owned fixture path; no shell.
    assert identity["git_revision"] == actual
    assert identity["dirty_source"] is False
    module.write_text("VALUE = 2\n")
    changed = measure_baseline.source_identity(tmp_path)
    assert changed["git_revision"] == actual
    assert changed["dirty_source"] is True
    assert changed["source_sha256"] != identity["source_sha256"]


def test_archive_provenance_keeps_unverified_revision_separate(tmp_path, monkeypatch):
    (tmp_path / "sct").mkdir()
    (tmp_path / "sct" / "sample.py").write_text("VALUE = 1\n")
    monkeypatch.setenv("SCT_SOURCE_REVISION", "external-archive-claim")
    identity = measure_baseline.source_identity(tmp_path)
    assert identity["git_revision"] is None
    assert identity["dirty_source"] is None
    assert identity["declared_revision"] == "external-archive-claim"
    assert len(identity["source_sha256"]) == 64
