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
    assert "<IP_ADDRESS_0>" in result.lm_text
    assert "8.8.8.8" not in result.lm_text
    assert anon_map.deanonymize("<IP_ADDRESS_0>") == "8.8.8.8"
    assert len([e for e in anon_map.entries if e.entity_type == "IP_ADDRESS"]) == 1


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
    ("reversible", "<IP_ADDRESS_0>"),
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
