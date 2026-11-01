"""0.8.0 security review leftovers (research/0.8.0-security-review-findings.md): one regression per root cause."""

import time

import pytest

from sct import TextCleaner, TextCleanerConfig


def _cleaner(**kw):
    return TextCleaner(TextCleanerConfig.from_preset("training", check_ner_process=False, **kw))


# --- 1: a dash or dot look-alike in a URL host or mail domain (typeset text) ---

HOST_FORMS = ["‐", "‑", "‒", "–", "—", "﹣"]  # Pd dashes
DOT_FORMS = ["․", "﹒"]  # NFKC "."


@pytest.mark.parametrize("fold", [True, False])
@pytest.mark.parametrize("dash", HOST_FORMS)
def test_dash_in_url_host_masked(dash, fold):
    for host in (f"my{dash}site.example.com", f"my{dash}site.例え.jp"):
        out = _cleaner(fold_styled_characters=fold).process(f"see https://{host}/?token=SecretABC now").lm_text
        assert out == "see <URL> now", out


@pytest.mark.parametrize("dot", DOT_FORMS)
def test_dot_form_in_url_host_masked(dot):
    for url in (f"https://my{dot}example.com/?token=SecretABC", f"www{dot}example{dot}com/?token=SecretABC"):
        assert _cleaner().process(f"see {url} now").lm_text == "see <URL> now"


@pytest.mark.parametrize("sep", HOST_FORMS + DOT_FORMS)
def test_dash_or_dot_form_in_mail_domain_masked(sep):
    assert _cleaner().process(f"mail a@my{sep}site.example.com now").lm_text == "mail <EMAIL> now"


def test_scheme_url_safety_net_keeps_prose():
    cleaner = _cleaner(fold_styled_characters=False)
    # a scheme-led run the host grammar rejects is still a URL, up to a space or prose punctuation
    assert cleaner.process("(https://my‐site.com/x?t=Secret).").lm_text == "(<URL>)."
    assert cleaner.process("see http://intranet/page now").lm_text == "see <URL> now"
    # a bare scheme is no URL
    assert cleaner.process("use http:// or https://. ok").lm_text == "use http:// or https://. ok"


# --- 2: no regex scans a long token once per character (quadratic time) ---

def _seconds(cleaner, text):
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter()
        cleaner.process(text)
        best = min(best, time.perf_counter() - start)
    return best


@pytest.mark.parametrize("make", [
    lambda n: "http://" + "a1" * n + " ",  # URL authority
    lambda n: "http://x.com/" + "a1" * n + " ",  # URL run over a long path
    lambda n: "a1" * n,  # address-run scanners over one long word
    lambda n: "a-" * n,  # mail local part
    lambda n: "-" * 2 * n,  # a long rule line
    lambda n: "x@" + "a-" * n,
    lambda n: "á" * n,  # decomposed letters
    lambda n: "http://" + "a:" * n,  # URL user info
])
@pytest.mark.parametrize("fold", [True, False])
def test_long_tokens_near_linear(make, fold):
    cleaner = _cleaner(fold_styled_characters=fold)
    _seconds(cleaner, make(10))  # warm up
    small, large = _seconds(cleaner, make(2000)), _seconds(cleaner, make(8000))
    # 4x the length; linear is ~4x the time, quadratic 16x (generous for CI noise)
    assert large < 8 * small + 0.05, (small, large)


def test_linear_scanners_keep_matches():
    cleaner = _cleaner()
    # a URL glued after another one still folds, and user info with several colons is one URL
    assert cleaner.process("see https://a.jp,https：／／b.jp/?t=Secret now").lm_text == "see <URL>,<URL> now"
    assert cleaner.process("see https://u:p:q@x.com/?t=Secret now").lm_text == "see <URL> now"
    assert cleaner.process("mail jo-ann+tag@example.com now").lm_text == "mail <EMAIL> now"
