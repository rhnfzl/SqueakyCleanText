"""URLs: whole-host and query masking, non-Latin and IP hosts, brackets and punctuation at the edges."""

import dataclasses
from dataclasses import replace

import pytest

from sct import TextCleaner
from sct.adapters import process_json
from sct.config import TextCleanerConfig
from sct.presets import preset_config
from sct.streaming import WindowedStreamingRedactor
from sct.utils import constants


FAST = preset_config("training", check_ner_process=False)


# --- a query or fragment straight after the host is part of the URL ---
@pytest.mark.parametrize("url", [
    "https://example.com?patient=AliceSmith", "https://example.com#token=SecretABC",
])
def test_bare_host_query_and_fragment_are_replaced(url):
    assert TextCleaner(cfg=FAST).process(f"see {url} now").lm_text == "see <URL> now"


def test_bare_trailing_question_mark_stays_punctuation():
    assert TextCleaner(cfg=FAST).process("Seen https://example.com? Yes").lm_text == (
        "Seen <URL>? Yes")


# --- URL user info never runs across "/", "?" or markup to a later "@" ---
def test_url_user_info_stops_at_path_and_markup():
    url = constants.URL_REGEX
    assert url.sub("<URL>", "https://x.org/?to=alice@example.com&cc=bob%40example.net") == "<URL>"
    assert url.sub("<URL>", '<a href="https://x.org">alice@example.com</a>') == (
        '<a href="<URL>">alice@example.com</a>')
    assert url.sub("<URL>", "see https://user:pw@x.org/p and http://alice@x.org.") == (
        "see <URL> and <URL>.")


def test_percent_encoded_email_in_query_is_not_left_behind():
    cleaner = TextCleaner(cfg=TextCleanerConfig(check_ner_process=False))
    out = cleaner.process("https://x.org/?to=alice@example.com&cc=bob%40example.net").lm_text
    assert "bob" not in out and "example" not in out


# --- URLs with private, loopback or IPv6 hosts are replaced whole ---
@pytest.mark.parametrize("host", ["192.168.1.10", "10.0.0.5", "127.0.0.1", "172.16.0.1",
                                  "169.254.1.1", "[2001:db8::1]", "[::1]:8080"])
def test_url_with_ip_host_hides_query_values(host):
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False))
    text = f"see http://{host}/?email=alice%40example.com now"
    assert cleaner.process(text).lm_text == "see <URL> now"


# --- a ) ] } inside a URL does not end it; one around the URL stays outside ---
@pytest.mark.parametrize("text, expected", [
    ("https://example.com/?filter=(active)&email=alice%40example.com", "<URL>"),
    ("https://example.com/?q=[a]&token=abc%40def x", "<URL> x"),
    ("https://en.wikipedia.org/wiki/Foo_(bar)_baz x", "<URL> x"),
    ("see (https://x.com/a). ok", "see (<URL>). ok"),
    ("[l](https://x.com/a), b", "[l](<URL>), b"),
])
def test_url_interior_brackets_stay_in_url(text, expected):
    assert TextCleaner(cfg=FAST).process(text).lm_text == expected


MODES = ["placeholder", "synthetic", "reversible"]


def _clean_training(text, mode="placeholder", **kw):
    cfg = TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode, **kw)
    return TextCleaner(cfg).process(text).lm_text


# --- supplementary-plane, punycode and IPv4 hosts ---
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("text", [
    "info@𠮷野.jp", "x@example.xn--p1ai",
    "https://𠮷野.jp/?email=alice%40example.com",
    "https://example.xn--p1ai/?email=alice%40example.com",
    "http://10.0.0.0/?email=alice%40example.com",
    "http://10.0.0.255/?email=alice%40example.com",
])
def test_whole_host_is_matched(text, mode):
    out = _clean_training(f"see {text} now", mode)
    assert "alice" not in out and "p1ai" not in out and "𠮷" not in out and "10.0" not in out
    assert out.startswith("see ") and out.endswith(" now")


def test_url_ipv4_host_followed_by_label_is_host_name():
    assert _clean_training("http://10.0.0.1.example.com/x", "placeholder") == "<URL>"


# --- full-width ？ ＃ after a non-Latin host keep the query value inside the URL ---
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("url, secret", [("https://例え.テスト？email=alice%40example.com", "alice"),
                                         ("https://例え.テスト＃token=SecretABC", "SecretABC")])
def test_fullwidth_query_after_non_latin_host_masked(url, secret, mode):
    out = _clean_training(f"see {url} now", mode)
    assert secret not in out and "=" not in out


# --- a full-width dot between two non-Latin host labels; 2: a www URL with full-width ？ ＃ ---
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("url", ["https://例え．テスト/?token=SecretABC", "www.例え.テスト？token=SecretABC",
                                 "www.例え.テスト＃token=SecretABC", "www．例え．テスト？token=SecretABC"])
def test_non_latin_host_url_masked_whole(url, mode):
    out = _clean_training(f"go to {url} please", mode)
    assert "SecretABC" not in out and "例え" not in out and out.startswith("go to ") and out.endswith(" please")


def test_second_url_glued_in_cjk_prose_masked():
    out = _clean_training("https://www.example.com/見て！https://例え．テスト/a?token=SecretABC 終", "placeholder")
    assert "SecretABC" not in out and "例え" not in out


# --- an IPv4 host followed by a Unicode label is a host name, the whole URL is replaced ---
URLS = ["http://10.0.0.1.école.fr/a?token=SecretABC", "http://10.0.0.1.例子.中国/a?token=SecretABC",
        "http://10.0.0.1.𝒆𝒄𝒐𝒍𝒆.fr/a?token=SecretABC"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("url", URLS)
def test_ipv4_prefixed_unicode_host_url_replaced(url, mode):
    out = _clean_training(f"see {url} x", mode)
    assert "SecretABC" not in out and "token" not in out


def test_ipv4_url_before_sentence_end_kept():
    assert _clean_training("see http://10.0.0.1. Next") == "see <URL>. Next"


@pytest.mark.parametrize("mode", MODES)
def test_ipv4_prefixed_native_digit_label_url_replaced(mode):
    out = _clean_training("see https://119.14.229.1௨3.pvbzn8iy.de/mu2g?q=SecretABC x", mode)
    assert "SecretABC" not in out


def _cleaner(mode="placeholder", **kw):
    return TextCleaner(TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode, **kw))


# --- CJK and full-width prose punctuation glued after a bare URL host ends the host ---
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


# --- a layout hint (ZWSP, word joiner) inside a Unicode URL host drops, as in a mail domain ---
HINTED = ["See https://例​え.jp/?token=SecretABC now", "See https://例⁠え.jp/?token=SecretABC now",
          "See www.例​え.jp/?token=SecretABC now"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("text", HINTED)
def test_hinted_unicode_host_url_masked(text, mode):
    out = _cleaner(mode).process(text).lm_text
    assert "SecretABC" not in out and "例" not in out, out


def test_hinted_unicode_host_json_and_stream():
    cleaner = _cleaner()
    assert cleaner.process(HINTED[0]).lm_text == "See <URL> now"
    assert "SecretABC" not in str(process_json(cleaner, {"note": HINTED[0]}).value)
    for pad in range(0, 30):
        stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=40, overlap_chars=16)
        out = stream.feed("x " * pad + HINTED[0]) + stream.finish()
        assert "SecretABC" not in out, (pad, out)


def test_join_controls_stay_in_url_host():
    # ZWNJ spells the word (Persian), so it is part of the host
    text = "https://می‌خواهم.ir/?t=SecretABC"
    assert "SecretABC" not in _cleaner().process(text).lm_text


def _stream_safe(cleaner, text, secret):
    for pad in range(0, 30):
        stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=80, overlap_chars=16)  # holds the URL
        out = stream.feed("x " * pad + text) + stream.finish()
        assert secret not in out, (pad, out)


# --- a full-width @ after punctuation in URL user info folds with the authority ---
USERINFO_URLS = ["https://alice:pass!＠example.com/?token=SecretABC",
                 "see https://alice:p4ss!＠example.com/x?token=SecretABC now",
        "see https://alice:p4ss！＠example.com/x?token=SecretABC now"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("text", USERINFO_URLS)
def test_userinfo_styled_at_after_punct_masked(text, mode):
    out = _cleaner(mode).process(text).lm_text
    assert "SecretABC" not in out and "pass" not in out and "p4ss" not in out, out


def test_userinfo_styled_at_json_and_stream():
    cleaner = _cleaner()
    assert cleaner.process(USERINFO_URLS[1]).lm_text == "see <URL> now"
    assert "SecretABC" not in str(process_json(cleaner, {"note": USERINFO_URLS[0]}).value)
    _stream_safe(cleaner, USERINFO_URLS[1], "SecretABC")


DEFAULT = TextCleanerConfig(check_ner_process=False)


# --- a ) ] } before a file extension stays in the URL, so the query after it is masked ---
@pytest.mark.parametrize("cfg", [FAST, DEFAULT], ids=["training", "default"])
@pytest.mark.parametrize("text, expected", [
    ("Download https://files.example.com/report(1).pdf?email=alice%40example.com now.",
     "Download <URL> now."),
    ("https://en.wikipedia.org/wiki/Foo_(bar).html?token=abc123 x", "<URL> x"),
    ("https://x.com/?f=[a].b&email=bob%40x.org x", "<URL> x"),
    ("see (https://x.com/a). ok", "see (<URL>). ok"),
    ("Link: https://x.com/x_(y).", "Link: <URL>)."),
])
def test_closing_bracket_before_extension_stays_in_url(cfg, text, expected):
    cleaner = TextCleaner(cfg=cfg)
    out = cleaner.process(text).lm_text
    assert "alice" not in out and "bob" not in out and "abc123" not in out
    if cfg is FAST:
        assert out == expected


# --- a run of closing brackets followed by more URL text stays in the URL ---
@pytest.mark.parametrize("cfg", [FAST, DEFAULT], ids=["training", "default"])
@pytest.mark.parametrize("mode", ["placeholder", "reversible", "synthetic"])
@pytest.mark.parametrize("text, expected", [
    ("Open https://portal.example.com/(S(lit3py55t21z5v55vlm25s55))/orderform.aspx?email=alice%40example.com now.",
     "Open <URL> now."),
    ("See https://x.example.com/a/((b))/c?email=alice%40b.com now.", "See <URL> now."),
    ("Go https://x.example.com/?q=f(a(b))&email=alice%40b.com now.", "Go <URL> now."),
    ("Read (https://x.example.com/a(b)) now.", "Read (<URL>)) now."),
])
def test_closing_bracket_run_stays_in_url(cfg, mode, text, expected):
    cleaner = TextCleaner(cfg=replace(cfg, replacement_mode=mode))
    out = cleaner.process(text).lm_text
    assert "alice" not in out and "lit3py" not in out
    if cfg is FAST and mode == "placeholder":
        assert out == expected


# --- a closing bracket followed by a URL sub-delimiter (; , $ ...) and more URL text stays in the URL ---
@pytest.mark.parametrize("cfg", [FAST, DEFAULT], ids=["training", "default"])
@pytest.mark.parametrize("mode", ["placeholder", "reversible", "synthetic"])
@pytest.mark.parametrize("text", [
    "See https://example.com/?q=(report);email=alice%40example.com now.",
    "See https://example.com/a(b);jsessionid=SecretABC now.",
    "See https://example.com/a(b),token=SecretABC now.",
    "See https://example.com/a(b)$token=SecretABC now.",
])
def test_bracket_before_sub_delimiter_stays_in_url(cfg, mode, text):
    out = TextCleaner(cfg=dataclasses.replace(cfg, replacement_mode=mode)).process(text).lm_text
    assert "alice" not in out and "Secret" not in out
    if cfg is FAST and mode == "placeholder":
        assert out == "See <URL> now."


@pytest.mark.parametrize("text, expected", [
    ("(see https://x.example.com/a), and more.", "(see <URL>), and more."),
    ("(https://x.example.com/a); then", "(<URL>); then"),
    ("(https://x.example.com/a)'s page", "(<URL>)'s page"),
])
def test_prose_punctuation_after_bracketed_url_stays(text, expected):
    assert TextCleaner(cfg=FAST).process(text).lm_text == expected


# --- a fully qualified host's final dot does not end the URL before its path or query ---
@pytest.mark.parametrize("url", [
    "https://portal.example.com./?email=alice%40example.com",
    "www.example.com./path?u=bob%40x.org",
    "http://10.0.0.1./?e=alice%40b.com",
    "http://localhost./?e=alice%40b.com",
    "https://example.com.:8080/a?e=alice",
    "https://example.com.#frag=alice",
])
def test_url_with_final_host_dot_is_whole(url):
    cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False))
    text = f"Open {url} now."
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=64, overlap_chars=60)  # holds the URL
    outputs = [cleaner.process(text).lm_text, cleaner.process_batch([text])[0].lm_text,
               stream.feed(text) + stream.finish()]
    assert outputs == ["Open <URL> now."] * 3


@pytest.mark.parametrize("text", ["Visit https://example.com. Next one.", "Is it https://example.com.? Yes."])
def test_sentence_period_after_url_stays(text):
    out = TextCleaner(TextCleanerConfig(check_ner_process=False)).process(text).lm_text
    assert out == text.replace("https://example.com", "<URL>")


def test_scheme_with_mixed_width_slashes_folds():
    assert "yuek6" not in _clean_training("see ｈtｔp：/／8cmca．net／yuek6？email=x now")


def test_url_glued_after_thai_letters():
    assert "iyfmyxg1fn" not in _clean_training("ไปที่เว็บhttps://ldp3.example.nl/pl5wj?id=iyfmyxg1fn ครับ")


def test_url_path_ends_at_cjk_punctuation():
    out = _clean_training("見 https://example.de/a?q=zz（備考）！06 08 66 90 80 です")
    assert "（備考）" in out and "66 90" not in out
