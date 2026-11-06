"""Regression tests for 0.7.1 regex fixes: digit boundaries, IPs, phones, step order."""

import warnings

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.browser import build_browser_manifest
from sct.utils import constants
from sct.utils.contact import ProcessContacts
from sct.utils.stopwords import ProcessStopwords

CONTACTS = ProcessContacts()


def _cleaner(**overrides):
    cfg = dict(
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
        check_to_ascii_unicode=False,
    )
    cfg.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TextCleaner(cfg=TextCleanerConfig(**cfg))


@pytest.mark.parametrize("text", [
    "फोन9876543210पर",   # Devanagari, both sides
    "কল9876543210",      # Bengali
    "电话9876543210",     # CJK
    "அழை9876543210",     # Tamil (combining mark before the digits)
    "اتصل9876543210",    # Arabic
    "Tel9876543210",     # Latin
])
def test_phone_glued_to_letters_any_script(text):
    assert "9876543210" not in CONTACTS.replace_phone_numbers(text)


@pytest.mark.parametrize("text, expected", [
    ("(tel +31 6 1234 5678)", "(tel <PHONE>)"),
    ("Ring +91 98765 43210).", "Ring <PHONE>)."),
    ("+44 (0)20 7946 0958", "<PHONE>"),
    ("+49 (30) 1234567", "<PHONE>"),
])
def test_international_phone_keeps_closing_paren_of_sentence(text, expected):
    assert CONTACTS.replace_phone_numbers(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("राशि5000रुपये", "राशि<NUMBER>रुपये"),
    ("金额5000元", "金额<NUMBER>元"),
])
def test_number_glued_to_letters_any_script(text, expected):
    assert CONTACTS.replace_numbers(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("वर्ष2023में", "वर्ष<YEAR>में"),
    ("年份2023年", "年份<YEAR>年"),
    ("FY2023", "FY<YEAR>"),
    ("12023", "12023"),  # still not inside a longer number
])
def test_year_glued_to_letters_any_script(text, expected):
    assert constants.YEAR_REGEX.sub("<YEAR>", text) == expected


@pytest.mark.parametrize("phone", [
    "+34 612 345 678",
    "+91 98765 43210",
    "98765 43210",
    "098765-43210",
    "+31 6 1234 2019",
    "+49 30 12345678",
])
def test_international_and_indian_phones(phone):
    assert CONTACTS.replace_phone_numbers(f"Call {phone} now") == "Call <PHONE> now"


@pytest.mark.parametrize("text", [
    "In 1999 2000 2001",
    "from 1999-2000",
    "raised 50000 50000",
    "price 12345 and 1,234,567",
    "zip 1012 AB",
])
def test_non_phones_not_detected(text):
    assert CONTACTS.replace_phone_numbers(text) == text


@pytest.mark.parametrize("text, expected", [
    ("Server 2001:4860:4860::8888 down", "Server <IP> down"),
    ("Host 192.168.1.20 and 8.8.8.8 seen", "Host <IP> and <IP> seen"),
    ("Call +31 6 1234 2019", "Call <PHONE>"),
    ("loopback ::1 and fe80::1", "loopback <IP> and <IP>"),
    ("In 1999 2000 2001 we grew", "In <YEAR> <YEAR> <YEAR> we grew"),
])
def test_pipeline_ips_then_phones_then_years(text, expected):
    assert _cleaner().process(text)[0] == expected


@pytest.mark.parametrize("text", ["std::vector", "at 12:30:45", "v1.2.3.4.5", "256.1.1.1"])
def test_ip_regex_skips_non_ips(text):
    assert CONTACTS.replace_ips(text) == text


def test_ips_public_only_keeps_private_addresses():
    text = "Host 192.168.1.20 and 8.8.8.8 and 2001:4860:4860::8888"
    assert CONTACTS.replace_ips(text, public_only=True) == (
        "Host 192.168.1.20 and <IP> and <IP>"
    )
    cleaner = _cleaner(
        replace_ips_public_only=True,
        check_replace_numbers=False,
        check_remove_isolated_special_symbols=False,
    )
    assert cleaner.process("Host 10.0.0.1 and 8.8.8.8")[0] == "Host 10.0.0.1 and <IP>"


def test_ip_step_can_be_disabled():
    cleaner = _cleaner(
        check_replace_ips=False,
        check_replace_numbers=False,
        check_replace_years=False,
        check_remove_isolated_special_symbols=False,
    )
    assert cleaner.process("Host 8.8.8.8")[0] == "Host 8.8.8.8"


def test_ip_reversible_mode_restores_and_respects_public_only():
    cleaner = _cleaner(replacement_mode="reversible", replace_ips_public_only=True)
    result = cleaner.process("Host 10.0.0.1 and 8.8.8.8")
    anon_map = result.metadata["anon_map"]
    assert "<IP_0>" in result.lm_text
    assert "8.8.8.8" not in result.lm_text
    assert anon_map.deanonymize("<IP_0>") == "8.8.8.8"
    assert len([e for e in anon_map.entries if e.entity_type == "IP"]) == 1


def test_ip_synthetic_mode_replaces_address():
    pytest.importorskip("faker")
    cleaner = _cleaner(
        replacement_mode="synthetic",
        check_replace_numbers=False,
        check_replace_years=False,
        check_replace_phone_numbers=False,
    )
    assert "8.8.8.8" not in cleaner.process("Host 8.8.8.8 seen")[0]


def test_ip_legacy_module_globals():
    from sct import config
    saved = (config.CHECK_NER_PROCESS, config.CHECK_REPLACE_IPS,
             config.REPLACE_IPS_PUBLIC_ONLY, config.REPLACE_WITH_IPS)
    try:
        config.CHECK_NER_PROCESS = False
        config.CHECK_REPLACE_IPS = False
        config.REPLACE_IPS_PUBLIC_ONLY = True
        config.REPLACE_WITH_IPS = "[IP]"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = TextCleaner().cfg
        assert cfg.check_replace_ips is False
        assert cfg.replace_ips_public_only is True
        assert cfg.replace_with_ips == "[IP]"
    finally:
        (config.CHECK_NER_PROCESS, config.CHECK_REPLACE_IPS,
         config.REPLACE_IPS_PUBLIC_ONLY, config.REPLACE_WITH_IPS) = saved


def test_browser_rule_order_matches_python_pipeline():
    manifest = build_browser_manifest(TextCleanerConfig())
    order = manifest["rule_order"]
    assert order.index("ip") < order.index("date") < order.index("phone") < order.index("year")
    assert manifest["rules"]["ip"]["pattern"] == constants.IP_REGEX.pattern
    with pytest.raises(ValueError, match="replace_ips_public_only"):
        build_browser_manifest(TextCleanerConfig(replace_ips_public_only=True))


def test_stopwords_unsupported_language_does_not_crash():
    sw = ProcessStopwords(supported_languages=frozenset({"ENGLISH", "THAI"}))
    assert sw.get_stop_words("THAI") == set()


# --- Review round 1 regressions ---

@pytest.mark.parametrize("ip", [
    "::abcd:192.0.2.1",
    "1:2:3:4:5::192.0.2.1",
    "1:2::3:192.0.2.1",
    "::ffff:0:192.0.2.1",
    "2001:db8::",
    "1::8",
    "1:2:3:4:5:6:7::",
])
@pytest.mark.parametrize("mode", ["placeholder", "synthetic", "reversible"])
def test_ipv6_replaced_whole_in_every_mode(ip, mode):
    cleaner = _cleaner(replacement_mode=mode)
    result = cleaner.process(f"host {ip} up")
    assert ip not in result.lm_text
    leftover = constants.IP_REGEX.sub("", result.lm_text)
    assert ":" not in leftover  # no IPv6 prefix or IPv4 tail left behind
    if mode == "reversible":
        assert result.metadata["anon_map"].deanonymize(result.lm_text) == f"host {ip} up"


@pytest.mark.parametrize("text", ["1:2:3:4:5:6::7:8", "a::b::c", "x :: y"])
def test_ip_regex_rejects_invalid_ipv6_like_python(text):
    # The browser uses this regex without ipaddress validation, so it must be exact.
    assert CONTACTS.replace_ips(text) == text
    assert constants.IP_REGEX.search(text) is None


def test_emitted_tokens_with_digits_survive_later_steps():
    cleaner = _cleaner(
        replace_with_email="<EMAIL1>",
        replace_with_url="<URL2>",
        replace_with_html="<HTML3>",
        replace_with_ips="<IP4>",
        replace_with_phone_numbers="<PHONE2023>",
        replace_with_years="<YEAR5>",
        check_replace_dates=True,
        replace_with_dates="<DATE2024>",
    )
    text = "<b>x</b> a@b.com https://x.io 8.8.8.8 +1-202-555-0147 1999 2023-01-01"
    assert cleaner.process(text).lm_text == (
        "<HTML3>x<HTML3> <EMAIL1> <URL2> <IP4> <PHONE2023> <YEAR5> <DATE2024>"
    )


def test_synthetic_ip_and_phone_not_rewritten_by_number_step():
    out = _cleaner(replacement_mode="synthetic").process(
        "Host 8.8.8.8 or 2001:4860::8888"
    ).lm_text
    assert len(constants.IP_REGEX.findall(out)) == 2


@pytest.mark.parametrize("mode, public", [
    ("placeholder", "<IP>"),
    ("synthetic", None),
    ("reversible", "<IP_0>"),
])
def test_public_only_kept_ips_survive_default_later_steps(mode, public):
    text = "Host 10.0.0.1 and fd00::1 and 8.8.8.8 in 2023"
    out = _cleaner(replacement_mode=mode, replace_ips_public_only=True).process(text).lm_text
    assert out.startswith("Host 10.0.0.1 and fd00::1 and ")
    assert "8.8.8.8" not in out
    if public:
        assert public in out


def test_browser_export_rejects_digit_tokens():
    with pytest.raises(ValueError, match="without digits"):
        build_browser_manifest(TextCleanerConfig(replace_with_email="<EMAIL1>"))


@pytest.mark.parametrize("text, ip", [
    ("client_ip:fd12:3456:789a::1", "fd12:3456:789a::1"),
    ("src:fe80::1", "fe80::1"),
    ("地址fe80::abcd", "fe80::abcd"),
    ("fe80::abcd地址", "fe80::abcd"),
])
@pytest.mark.parametrize("mode", ["placeholder", "synthetic", "reversible"])
@pytest.mark.parametrize("numbers", [True, False])
def test_ipv6_beside_labels_and_any_script_replaced(text, ip, mode, numbers):
    out = _cleaner(replacement_mode=mode, check_replace_numbers=numbers).process(text).lm_text
    assert ip not in out
    assert not any(part in out for part in ("fd12", "3456", "789a", "fe80", "abcd", "<NUMBER>"))


@pytest.mark.parametrize("text", ["std::vector", "Foo::Bar", "1:2:3:4:5:6:7:8:9"])
def test_ip_regex_keeps_ascii_glued_non_addresses(text):
    assert constants.IP_REGEX.search(text) is None


@pytest.mark.parametrize("text, expected", [
    ("mail alice@example.com<br> now", "mail <EMAIL><HTML> now"),
    ("see https://example.com<br> now", "see <URL><HTML> now"),
])
def test_contacts_beside_protected_html_token(text, expected):
    out = _cleaner().process(text).lm_text
    assert out == expected


@pytest.mark.parametrize("mode", ["placeholder", "synthetic"])
def test_fuzzy_date_token_not_rewritten_by_year_step(mode):
    out = _cleaner(
        replacement_mode=mode, check_fuzzy_replace_dates=True, replace_with_dates="<DATE2024>",
    ).process("on 12 Januray 2023 we met").lm_text
    assert "<DATE2024>" in out
    assert "2023" not in out


def test_browser_export_rejects_currency_tokens():
    with pytest.raises(ValueError, match="without currency symbols"):
        build_browser_manifest(TextCleanerConfig(replace_with_email="$EMAIL"))


@pytest.mark.parametrize("field, token, later", [
    ("replace_with_html", "alice@example.com", "email"),
    ("replace_with_html", "https://example.com", "url"),
    ("replace_with_url", "alice@example.com", "email"),
])
def test_browser_export_rejects_tokens_later_rules_rewrite(field, token, later):
    with pytest.raises(ValueError, match=f"later {later} rule"):
        build_browser_manifest(TextCleanerConfig(**{field: token}))
    # A disabled later rule cannot rewrite the token.
    off = {"email": "check_replace_emails", "url": "check_replace_urls"}[later]
    build_browser_manifest(TextCleanerConfig(**{field: token, off: False}))


@pytest.mark.parametrize("text", [
    "We use GPT-4o and GPT4.",
    "Qwen3-8B and Mistral7B.",
    "Phi4, H2O, mp3, ΑΒΓ2, Яндекс3",
])
def test_numbers_in_spaced_script_words_kept_by_default(text):
    assert _cleaner(check_replace_years=False).process(text).lm_text == text


def test_numbers_in_words_flag_replaces_glued_digits():
    out = _cleaner(replace_numbers_in_words=True, replace_numbers_mode="all").process("GPT4 and mp3").lm_text
    assert out == "GPT<NUMBER> and mp<NUMBER>"


def test_numbers_in_words_default_still_replaces_spaceless_scripts():
    assert CONTACTS.replace_numbers("金额5000元") == "金额<NUMBER>元"
    assert CONTACTS.replace_numbers("Llama-3.1-70B") == "Llama-<NUMBER>-70B"  # same as 0.7.0
    assert CONTACTS.replace_numbers("total EUR4500, INR5000") == "total EUR<NUMBER>, INR<NUMBER>"
    assert CONTACTS.replace_numbers("XEUR4500") == "XEUR4500"


def test_numbers_in_words_legacy_global(monkeypatch):
    from sct import config
    monkeypatch.setattr(config, "REPLACE_NUMBERS_IN_WORDS", True)
    assert config._config_from_module_globals().replace_numbers_in_words is True


def test_browser_number_rule_follows_flag():
    default = build_browser_manifest(TextCleanerConfig(check_ner_process=False))
    strict = build_browser_manifest(
        TextCleanerConfig(check_ner_process=False, replace_numbers_in_words=True))
    assert default != strict


# --- F3: IBAN / card checksums and identifier tokens (made-up values) ---

@pytest.mark.parametrize("value", [
    "NL91ABNA0417164300", "NL91 ABNA 0417 1643 00", "GB82 WEST 1234 5698 7654 32",
])
def test_valid_iban_is_one_iban_span(value):
    out = _cleaner().process(f"IBAN {value} ok").lm_text
    assert out == "IBAN <IBAN> ok"


def test_bad_iban_checksum_is_not_iban():
    out = _cleaner().process("IBAN NL92ABNA0417164300 ok").lm_text
    assert "<IBAN>" not in out


@pytest.mark.parametrize("value", ["4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111111"])
def test_luhn_card_is_one_card_span(value):
    assert _cleaner().process(f"Card {value} ok").lm_text == "Card <CARD> ok"


def test_bad_luhn_is_not_card_and_old_behaviour_reachable():
    assert "<CARD>" not in _cleaner().process("Card 4111111111111112").lm_text
    out = _cleaner(check_replace_iban_cards=False).process("Card 4111 1111 1111 1111").lm_text
    assert "<CARD>" not in out and "<PHONE>" in out


def test_iban_followed_by_code_word_keeps_checksum_prefix():
    out = _cleaner().process("BE68 5390 0754 7034 ABCD paid").lm_text
    assert out == "<IBAN> ABCD paid"


def test_identifier_rule_off_by_default_on_with_flag():
    text = "Licence RBCON68T08, ID XK4729QP31; word Model5 A1B2"
    assert _cleaner().process(text).lm_text.count("RBCON68T08") == 1
    out = _cleaner(check_replace_identifiers=True).process(text).lm_text
    assert out == "Licence <ID>, ID <ID>; word Model5 A1B2"


def test_new_types_reversible_and_counted():
    result = _cleaner(replacement_mode="reversible", check_replace_identifiers=True).process(
        "pay NL91ABNA0417164300 card 4111111111111111 ref RBCON68T08")
    anon_map = result.metadata["anon_map"]
    assert anon_map.deanonymize(result.lm_text) == "pay NL91ABNA0417164300 card 4111111111111111 ref RBCON68T08"
    assert {e.entity_type for e in anon_map.entries} == {"IBAN", "CARD", "ID"}
    assert {"IBAN", "CARD", "ID"} <= set(result.signals["pii_types"])


def test_new_types_synthetic():
    out = _cleaner(replacement_mode="synthetic", check_replace_identifiers=True).process(
        "pay NL91ABNA0417164300 card 4111111111111111 ref RBCON68T08").lm_text
    assert "NL91ABNA0417164300" not in out and "4111111111111111" not in out and "RBCON68T08" not in out


@pytest.mark.parametrize("text, expected", [
    ("Contact John Smith <john.smith@example.com> today.", "Contact John Smith <<EMAIL>> today."),
    # a lone "<" is dropped as a symbol, not as markup
    ("If a < b then mail <x@y.org> please", "If a b then mail <<EMAIL>> please"),
    ("see <https://example.com/path> now", "see <<URL>> now"),
    ("x <3 y > 2", "x <3 y 2"),
    ("a <b>bold</b> <br/> <!-- note --> end", "a <HTML>bold<HTML> <HTML> <HTML> end"),
    ('<!DOCTYPE html><a\nhref="mailto:x@y.org">hi</a>', "<HTML><HTML>hi<HTML>"),
])
@pytest.mark.parametrize("mode", ["placeholder", "reversible"])
def test_html_rule_reads_only_tag_shapes(text, expected, mode):
    out = _cleaner(replacement_mode=mode, check_replace_numbers=False,
                   check_remove_isolated_letters=False).process(text).lm_text
    if mode == "placeholder":
        assert out == expected
    else:
        assert "<HTML" not in out or "<HTML" in expected
        assert "@" not in out and "https" not in out


def test_html_tag_strict_off_restores_any_angle_span():
    out = _cleaner(html_tag_strict=False).process("Contact Jo <jo@example.com> today.").lm_text
    assert out == "Contact Jo <HTML> today."
    js = build_browser_manifest(TextCleanerConfig(html_tag_strict=False))
    assert js["rules"]["html"]["pattern"].startswith("<.*?>")


@pytest.mark.parametrize("text, url", [
    ("Visit https://example.com/path. Next.", "https://example.com/path"),
    ("See https://example.com/a/b.html. Next", "https://example.com/a/b.html"),
    ("Go to https://x.com/p?q=1. Done", "https://x.com/p?q=1"),
    ("https://x.com/path, then", "https://x.com/path"),
    ("https://x.com/p; ok", "https://x.com/p"),
    ("https://x.com/a! wow", "https://x.com/a"),
    ("is it https://x.com/a? yes", "https://x.com/a"),
    ("at https://x.com/a: here", "https://x.com/a"),
    ("'https://x.com/a' said", "https://x.com/a"),
    ("(https://x.com/a.) ok", "https://x.com/a"),
    # punctuation inside the URL stays
    ("get https://x.com/a.html now", "https://x.com/a.html"),
    ("get https://x.com/p?a=1,2&b=3 now", "https://x.com/p?a=1,2&b=3"),
    ("get https://x.com/v1.2/x now", "https://x.com/v1.2/x"),
    ("get https://x.com/a;jsessionid=AB12 now", "https://x.com/a;jsessionid=AB12"),
    ("get https://x.com/a(b);jsessionid=1 now", "https://x.com/a(b);jsessionid=1"),
    ("get https://x.com/it's/x now", "https://x.com/it's/x"),
    ("get https://portal.example.com./?email=a%40b.org now", "https://portal.example.com./?email=a%40b.org"),
    ("Bare https://example.com. Next.", "https://example.com"),
])
def test_url_trailing_punct_stays_outside(text, url):
    assert constants.URL_REGEX.search(text).group() == url
    if text.startswith("'"):  # the cleaner drops the quotes themselves; the regex check above is the point
        return
    out = _cleaner(check_replace_numbers=False, check_remove_isolated_letters=False).process(text).lm_text
    assert out == text.replace(url, "<URL>")


def test_url_trailing_punct_outside_off_restores_080():
    text = "Visit https://example.com/path. Next."
    assert constants.URL_REGEX_LEGACY.search(text).group() == "https://example.com/path."
    assert _cleaner(url_trailing_punct_outside=False).process(text).lm_text == "Visit <URL> Next."
    js = build_browser_manifest(TextCleanerConfig(url_trailing_punct_outside=False))
    assert js["rules"]["url"]["pattern"] == constants.URL_REGEX_LEGACY.pattern


# F12: EMAIL_GATE is a cheap pre-check for the email regex's separator; it must never skip a match.
@pytest.mark.parametrize("text", [
    "a@b.com", "x[AT]y.org", "foo{at}bar.de", "mail (at) host", "用户@例子.广告", "anna!@example.com",
    "Contact John Smith <john.smith@example.com> today.", "<x@y.org>", "joe<AT>mail.example.net",
])
def test_email_gate_passes_every_regex_match(text):
    for match in constants.EMAIL_REGEX.finditer(text):
        assert constants.EMAIL_GATE.search(match.group())


class _NoSub:
    def sub(self, *args, **kwargs):
        raise AssertionError("email regex ran on text with no separator")


@pytest.mark.parametrize("reversible", [False, True])
def test_email_regex_skipped_without_separator(monkeypatch, reversible):
    cleaner = _cleaner(replacement_mode="reversible" if reversible else "placeholder")
    monkeypatch.setattr(constants, "EMAIL_REGEX", _NoSub())
    text = "A long paragraph with no address in it, only words."
    assert CONTACTS.replace_emails(text) == text
    assert CONTACTS.replace_emails_with_fn(text, lambda s: "<EMAIL>") == text
    cleaner.process(text)
