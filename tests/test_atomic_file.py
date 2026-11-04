import errno
import os
import stat

import pytest

from sct.utils import atomic_file
from sct.utils.atomic_file import write_text_atomic


def test_replaces_content_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "out.json"
    target.write_text("old", encoding="utf-8")
    write_text_atomic(target, "new é")
    assert target.read_text(encoding="utf-8") == "new é"
    assert os.listdir(tmp_path) == ["out.json"]


def test_a_failed_write_keeps_the_old_file(tmp_path, monkeypatch):
    target = tmp_path / "out.json"
    target.write_text("old", encoding="utf-8")

    def fail(*_):
        raise OSError("disk full")
    monkeypatch.setattr(atomic_file.os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        write_text_atomic(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
    assert os.listdir(tmp_path) == ["out.json"]


def test_keeps_the_mode_of_an_existing_file(tmp_path):
    target = tmp_path / "out.json"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o640)
    write_text_atomic(target, "new")
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_new_file_mode_matches_write_text(tmp_path):
    plain, atomic = tmp_path / "plain", tmp_path / "atomic"
    plain.write_text("x", encoding="utf-8")
    write_text_atomic(atomic, "x")
    assert stat.S_IMODE(atomic.stat().st_mode) == stat.S_IMODE(plain.stat().st_mode)


def test_a_symlink_keeps_pointing_at_the_replaced_file(tmp_path):
    real, link = tmp_path / "real", tmp_path / "link"
    real.write_text("old", encoding="utf-8")
    link.symlink_to(real)
    write_text_atomic(link, "new")
    assert link.is_symlink()
    assert real.read_text(encoding="utf-8") == "new"


def test_a_missing_parent_folder_still_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_text_atomic(tmp_path / "no" / "out.json", "x")


def _write_to_dev_stdout_bound_to(descriptor):
    saved = os.dup(1)
    os.dup2(descriptor, 1)
    try:
        write_text_atomic("/dev/stdout", "report")
    finally:
        os.dup2(saved, 1)
        os.close(saved)


needs_dev_stdout = pytest.mark.skipif(not os.path.exists("/dev/stdout"), reason="needs /dev/stdout")


@needs_dev_stdout
def test_dev_stdout_redirected_to_a_file_is_written_through(tmp_path):
    out = tmp_path / "report.json"
    with out.open("w", encoding="utf-8") as handle:
        _write_to_dev_stdout_bound_to(handle.fileno())
    assert out.read_text(encoding="utf-8") == "report"
    assert os.listdir(tmp_path) == ["report.json"]


@needs_dev_stdout
def test_dev_stdout_piped_is_written_through():
    reader, writer = os.pipe()
    try:
        _write_to_dev_stdout_bound_to(writer)
        assert os.read(reader, 100) == b"report"
    finally:
        os.close(reader)
        os.close(writer)


def test_a_writable_file_in_a_read_only_folder_is_still_written(tmp_path):
    target = tmp_path / "out.json"
    target.write_text("old", encoding="utf-8")
    tmp_path.chmod(0o555)
    try:
        write_text_atomic(target, "new")
    finally:
        tmp_path.chmod(0o755)
    assert target.read_text(encoding="utf-8") == "new"


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file modes")
def test_a_read_only_file_still_raises_and_stays(tmp_path):
    target = tmp_path / "out.json"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o444)
    with pytest.raises(PermissionError):
        write_text_atomic(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
    assert os.listdir(tmp_path) == ["out.json"]


def test_a_name_too_long_for_the_temp_file_is_still_written(tmp_path):
    target = tmp_path / ("a" * 250)
    write_text_atomic(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


def test_a_full_disk_at_temp_creation_keeps_the_old_file(tmp_path, monkeypatch):
    target = tmp_path / "out.json"
    target.write_text("old", encoding="utf-8")

    def full(*_):
        raise OSError(errno.ENOSPC, "no space")
    monkeypatch.setattr(atomic_file.os, "open", full)
    with pytest.raises(OSError, match="no space"):
        write_text_atomic(target, "new")
    assert target.read_text(encoding="utf-8") == "old"
