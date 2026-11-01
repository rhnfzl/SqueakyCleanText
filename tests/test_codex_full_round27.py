"""Codex 0.8.0 full review, round 27: ids 1, 2, 3."""

import pickle

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.adapters import process_json
from sct.streaming import WindowedStreamingRedactor

MODES = ["placeholder", "synthetic", "reversible"]


def _cleaner(mode="placeholder", **kw):
    return TextCleaner(TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode, **kw))


def _stream_safe(cleaner, text, secret):
    for pad in range(0, 30):
        stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=40, overlap_chars=16)
        out = stream.feed("x " * pad + text) + stream.finish()
        assert secret not in out, (pad, out)


# --- 1: the IDNA middle dot between two l's (RFC 5892 A.3) stays inside a label ---

CATALAN = [("Escriu a info@col·legi.cat avui.", "legi"), ("Escriu a COL·LEGI@example.com avui.", "COL"),
           ("Vegeu https://www.col·legi.cat/?token=SecretABC avui.", "SecretABC")]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("text,secret", CATALAN)
def test_catalan_middle_dot_masked(text, secret, mode):
    out = _cleaner(mode).process(text).lm_text
    assert secret not in out and "·" not in out, out


def test_catalan_middle_dot_json_stream():
    cleaner = _cleaner()
    assert cleaner.process(CATALAN[0][0]).lm_text == "Escriu a <EMAIL> avui."
    assert "SecretABC" not in str(process_json(cleaner, {"note": CATALAN[2][0]}).value)
    _stream_safe(cleaner, CATALAN[2][0], "SecretABC")


def test_middle_dot_outside_l_l_ends_label():
    # elsewhere the middle dot is punctuation: an address ends before it
    assert _cleaner().process("info@example.cat·Bob").lm_text == "<EMAIL>·Bob"


# --- 2, 3: a fold run takes every local-part character the email grammar takes, typeset dashes too ---

MAILS = ["Mail anna.smith!＠example.com now", "Mail josé＿smith.work@example.com now",
         "Mail john.smith‑billing@example.com now", "Mail john.smith‐billing@example.com now",
         "Mail anna.smith!﹫example.com now", "Mail anna.smith\u00ad＊＠example.com now",
         "Mail anna\u2024\u00adsmith﹪﹫example.com now"]


@pytest.mark.parametrize("fold", [True, False])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("text", MAILS[:2])
def test_local_part_punctuation_folds(text, mode, fold):
    out = _cleaner(mode, fold_styled_characters=fold).process(text).lm_text
    assert "smith" not in out and "anna" not in out and "josé" not in out, out


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("text", MAILS[2:])
def test_local_part_styled_forms_fold(text, mode):
    out = _cleaner(mode).process(text).lm_text
    assert "smith" not in out and "anna" not in out, out
    assert out.startswith("Mail ") and out.endswith(" now"), out


def test_local_part_json_stream_pickle_batch():
    cleaner = _cleaner()
    for text in MAILS:
        assert cleaner.process(text).lm_text == "Mail <EMAIL> now", text
        assert "smith" not in str(process_json(cleaner, {"note": text}).value)
        _stream_safe(cleaner, text, "smith")
    cfg = pickle.loads(pickle.dumps(cleaner.cfg))  # noqa: S301  # our own config, round-tripped
    assert [r.lm_text for r in TextCleaner(cfg).process_batch(MAILS)] == ["Mail <EMAIL> now"] * len(MAILS)


def test_prose_punctuation_without_at_keeps_width():
    assert _cleaner().process("你好！世界．テスト＿です").lm_text == "你好！世界．テスト＿です"
