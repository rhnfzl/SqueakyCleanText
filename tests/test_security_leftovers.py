"""0.8.0 security review leftovers (research/0.8.0-security-review-findings.md): one regression per root cause."""

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
