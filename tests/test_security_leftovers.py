"""0.8.0 security review leftovers (research/0.8.0-security-review-findings.md): one regression per root cause."""

import pytest

from sct import TextCleaner, TextCleanerConfig
from tests.helpers import assert_near_linear


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

@pytest.mark.parametrize("make", [
    lambda n: "http://" + "a1" * n + " ",  # URL authority
    lambda n: "http://x.com/" + "a1" * n + " ",  # URL run over a long path
    lambda n: "a1" * n,  # address-run scanners over one long word
    lambda n: "a-" * n,  # mail local part
    lambda n: "-" * 2 * n,  # a long rule line
    lambda n: "x@" + "a-" * n,
    lambda n: "á" * n,  # decomposed letters
    lambda n: "http://" + "a:" * n,  # URL user info
    lambda n: "a" + "\uff0d" * 2 * n + " ",  # styled punctuation run with no letter or digit after it
    lambda n: "a" + "-" * n + "\uff0e" + "\uff0f" * n + " ",
])
@pytest.mark.parametrize("fold", [True, False])
def test_long_tokens_near_linear(make, fold):
    cleaner = _cleaner(fold_styled_characters=fold)
    cleaner.process(make(10))  # warm up
    assert_near_linear(cleaner.process, make, 8000, factor=4)


def test_linear_scanners_keep_matches():
    cleaner = _cleaner()
    # a URL glued after another one still folds, and user info with several colons is one URL
    assert cleaner.process("see https://a.jp,https：／／b.jp/?t=Secret now").lm_text == "see <URL>,<URL> now"
    assert cleaner.process("see https://u:p:q@x.com/?t=Secret now").lm_text == "see <URL> now"
    assert cleaner.process("mail jo-ann+tag@example.com now").lm_text == "mail <EMAIL> now"


# --- 3: an IPv4 after a dot that ends a word (a CJK full stop folded to ".", or typed) ---

@pytest.mark.parametrize("fold", [True, False])
@pytest.mark.parametrize("text", ["x@ad.io銄黟。8.8.8.8夑", "see https://ad.io銄黟。8.8.8.8夑", "服务器.8.8.8.8",
                                  "host x.10.0.0.1"])
def test_ipv4_after_word_dot_masked(text, fold):
    out = _cleaner(fold_styled_characters=fold).process(text).lm_text
    assert "8.8.8.8" not in out and "10.0.0.1" not in out and "<IP>" in out, out


def test_ipv4_inside_longer_dotted_number_stays():
    assert _cleaner().process("v 1.2.3.4.5 x").lm_text == "v 1.2.3.4.5 x"


# --- 4: punctuation of any script (category P) ends a mail domain or URL host ---

@pytest.mark.parametrize("fold", [True, False])
@pytest.mark.parametrize("text, expected", [
    ("راسلني على ahmed.ali@example.com، أو", "راسلني على <EMAIL>، أو"),  # Arabic comma
    ("मेल a@b.com। फिर", "मेल <EMAIL>। फिर"),  # Devanagari danda
    ("«a@b.com» и", "«<EMAIL>» и"),
    ("see https://ab.com، x", "see <URL>، x"),
    ("メールa@例え.jp・次へ", "メール<EMAIL>・次へ"),
])
def test_punctuation_ends_domain(text, expected, fold):
    assert _cleaner(fold_styled_characters=fold).process(text).lm_text == expected
