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
