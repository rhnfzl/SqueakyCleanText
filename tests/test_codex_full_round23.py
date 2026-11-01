"""Codex 0.8.0 full review, round 23: id 1."""

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.adapters import process_json
from sct.streaming import WindowedStreamingRedactor

MODES = ["placeholder", "synthetic", "reversible"]


def _cleaner(mode="placeholder", **kw):
    return TextCleaner(TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode, **kw))


# --- 1: CJK and full-width prose punctuation glued after a bare URL host ends the host ---

TEXTS = ["https://example.com，電話：06 12 34 56 78 です", "https://例え.jp，電話：06 12 34 56 78 です",
         "www.example.com、電話：06 12 34 56 78"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("text", TEXTS)
def test_bare_host_glued_cjk_prose_keeps_phone(text, mode):
    out = _cleaner(mode).process(text).lm_text
    assert "12 34 56 78" not in out and "電話：" in out, out


def test_bare_host_glued_cjk_prose_kept_outside_url():
    assert _cleaner().process(TEXTS[0]).lm_text == "<URL>，電話：<PHONE> です"


def test_bare_host_glued_cjk_prose_json_and_stream():
    cleaner = _cleaner()
    assert "12 34 56 78" not in str(process_json(cleaner, {"note": TEXTS[0]}).value)
    for pad in range(0, 30):
        stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=40, overlap_chars=16)
        out = stream.feed("x " * pad + TEXTS[0]) + stream.finish()
        assert "12 34 56 78" not in out, (pad, out)


def test_sentence_dot_before_glued_scheme_stays():
    # 。 before a glued https:// ends the sentence; the first host does not run into the second URL
    out = _cleaner().process("see https://a.jp芢亱。https://b.jp/?q=SecretABC x").lm_text
    assert out == "see <URL>芢亱。<URL> x"


def test_circled_label_after_idna_dot_in_digit_host():
    out = _cleaner().process("see http://⑫𖭙.𖭗𖭗｡⑫｡⑮𖭔.цыэш．org/a?sid=SecretABC x").lm_text
    assert "SecretABC" not in out


def test_cjk_numbered_list_after_url_kept():
    assert _cleaner().process("詳細はhttps://example.com。①まず登録").lm_text == "詳細は<URL>。①まず登録"
