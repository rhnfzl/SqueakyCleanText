"""Codex 0.8.0 full review, round 38: id 1 (a Catalan l·l inside a styled address run)."""

import pickle
import time

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.adapters import process_json
from sct.streaming import WindowedStreamingRedactor
from sct.utils import normtext

MAILS = ["Mail info@col·legi。cat now", "Mail info@col·legi｡cat now", "Mail josé＿col·legi@example.cat now"]


def _cleaner(mode="placeholder", **kw):
    return TextCleaner(TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode, **kw))


# with fold_styled_characters off an ideographic full stop is prose (as without the middle dot); ＿ still folds
@pytest.mark.parametrize("mode", ["placeholder", "synthetic", "reversible"])
@pytest.mark.parametrize("text,fold", [(t, True) for t in MAILS] + [(MAILS[2], False)])
def test_middle_dot_with_styled_separator_masked(text, mode, fold):
    out = _cleaner(mode, fold_styled_characters=fold).process(text).lm_text
    assert "legi" not in out and "josé" not in out and "info" not in out, out
    assert out.startswith("Mail ") and out.endswith(" now"), out


def test_middle_dot_json_stream_pickle_batch():
    cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False))
    for text in MAILS:
        assert cleaner.process(text).lm_text == "Mail <EMAIL> now", text
        assert "legi" not in str(process_json(cleaner, {"note": text}).value)
        for pad in range(0, 30):
            stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=40, overlap_chars=16)
            out = stream.feed("x " * pad + text) + stream.finish()
            assert "legi" not in out, (pad, out)
    cfg = pickle.loads(pickle.dumps(cleaner.cfg))  # noqa: S301  # our own config, round-tripped
    assert [r.lm_text for r in TextCleaner(cfg).process_batch(MAILS)] == ["Mail <EMAIL> now"] * len(MAILS)


def test_middle_dot_outside_l_l_still_ends_run():
    assert _cleaner().process("Mail a·b@x。cat now").lm_text == "Mail a·<EMAIL> now"


@pytest.mark.parametrize("text", ["l·" * 5000 + "l", "l·l＿" * 5000, "a@" + "l·" * 5000, "l·l@" * 5000])
def test_middle_dot_runs_linear(text):
    start = time.perf_counter()
    normtext.fold_styled(text)
    normtext._fold_email_runs(text)
    assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize("text", ["Mail info＠ｃｏｌ­·ｌｅｇｉ．cat now", "Mail ana_col​·legi@example。cat now"])
def test_invisible_hint_beside_middle_dot_drops(text):
    # found by the fuzzer: a line-break hint between l and · is a copied layout hint like any other in the run
    assert _cleaner().process(text).lm_text == "Mail <EMAIL> now"
