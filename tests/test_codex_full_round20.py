"""Codex 0.8.0 full review, round 20: ids 1-2."""

import os
import sys
import threading
from datetime import datetime

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.streaming import WindowedStreamingRedactor
from sct.utils import synthetic

EMAILS = ["john@نامه\u200cای．ایران", "john@क्\u200dष．भारत", "مریم@نامه\u200cای．ایران"]


# --- 1: a join control (ZWNJ/ZWJ) inside an address does not stop its full-width dot folding ---


@pytest.mark.parametrize("fold", [True, False])
@pytest.mark.parametrize("mode", ["placeholder", "synthetic", "reversible"])
@pytest.mark.parametrize("email", EMAILS)
def test_join_control_email_with_fullwidth_dot_masked(email, mode, fold):
    cfg = TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode,
                                        fold_styled_characters=fold)
    out = TextCleaner(cfg).process(f"mail {email} now").lm_text
    assert "john" not in out and "ایران" not in out and "भारत" not in out
    assert out.startswith("mail ") and out.endswith(" now")


def test_join_control_email_masked_in_stream():
    cl = TextCleaner(TextCleanerConfig(check_ner_process=False))
    stream = WindowedStreamingRedactor(cl, max_buffer_chars=48, overlap_chars=12)
    doc = f"write to {EMAILS[0]} today"
    out = "".join(stream.feed(doc[i:i + 3]) for i in range(0, len(doc), 3)) + stream.finish()
    assert "john" not in out and "ایران" not in out


def test_zwnj_word_keeps_its_join_control():
    out = TextCleaner(TextCleanerConfig(check_ner_process=False)).process("نامه‌ای．ایران").lm_text
    assert out == "نامه‌ای．ایران"


# --- 2: a child forked while another thread holds the Faker clock gets the real clock back ---


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_forked_child_restores_held_clock():
    import faker.providers.date_time
    import faker.providers.ssn.zh_CN as ssn_zh

    def tax_id():
        replacer = synthetic.SyntheticReplacer(seed=42, locale="zh_CN", reference_date="2026-01-01")
        replacer.reset_cache("same document")
        return replacer.get_replacement("TAX_ID", "x")

    expected = tax_id()
    inside, release = threading.Event(), threading.Event()

    def hold():
        with synthetic._clock_at(datetime(1990, 6, 1), {ssn_zh, sys.modules[faker.providers.date_time.__name__]}):
            inside.set()
            release.wait()

    thread = threading.Thread(target=hold)
    thread.start()
    inside.wait()
    try:
        pid = os.fork()
        if pid == 0:  # pragma: no cover - child
            os._exit(0 if tax_id() == expected else 1)
    finally:
        release.set()
        thread.join()
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert tax_id() == expected


# --- 1 (fuzz): a join control inside the TLD label, as in any other host label ---


@pytest.mark.parametrize("email", ["ab@يت.ط‌ة‌ق", "ab@नाम.भा‍रत", "ab@يت．ط‌ة‌ق"])
def test_join_control_in_tld_masked(email):
    out = TextCleaner(TextCleanerConfig(check_ner_process=False)).process(f"mail {email} now").lm_text
    assert out == "mail <EMAIL> now"
