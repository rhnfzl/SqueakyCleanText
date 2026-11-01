"""Codex 0.8.0 full review, round 19: ids 1-2."""

import pytest

from sct import TextCleaner, TextCleanerConfig

MODES = ("placeholder", "synthetic", "reversible")


def _clean(text, mode):
    cfg = TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode)
    return TextCleaner(cfg).process(text).lm_text


# --- 1: a full-width dot between two non-Latin host labels; 2: a www URL with full-width ？ ＃ ---


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("url", ["https://例え．テスト/?token=SecretABC", "www.例え.テスト？token=SecretABC",
                                 "www.例え.テスト＃token=SecretABC", "www．例え．テスト？token=SecretABC"])
def test_non_latin_host_url_masked_whole(url, mode):
    out = _clean(f"go to {url} please", mode)
    assert "SecretABC" not in out and "例え" not in out and out.startswith("go to ") and out.endswith(" please")


def test_fullwidth_dot_in_prose_kept():
    assert _clean("例え．テスト", "placeholder") == "例え．テスト"


def test_second_url_glued_in_cjk_prose_masked():
    out = _clean("https://www.example.com/見て！https://例え．テスト/a?token=SecretABC 終", "placeholder")
    assert "SecretABC" not in out and "例え" not in out
