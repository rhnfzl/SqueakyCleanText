"""Email addresses in styled, full-width, non-Latin and invisible-character forms stay masked."""

import pickle
import time

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.adapters import process_json
from sct.presets import preset_config
from sct.streaming import WindowedStreamingRedactor
from sct.utils import constants, normtext
from sct.utils.normtext import fold_styled


FAST = preset_config("training", check_ner_process=False)


@pytest.fixture(scope="module")
def fast_cleaner():
    return TextCleaner(cfg=FAST)


# --- RFC punctuation in the local part left the address or a prefix visible ---
@pytest.mark.parametrize("text, expected", [
    ("Contact alice!@example.com", "Contact <EMAIL>"),
    ("Mail o'connor@example.com now", "Mail <EMAIL> now"),
    ("Mail jo&co@example.com now", "Mail <EMAIL> now"),
    ("'alice@example.com'", "'<EMAIL>'"),
    ("ends with dot. alice@example.com", "ends with dot. <EMAIL>"),
])
def test_email_local_part_punctuation_is_fully_masked(fast_cleaner, text, expected):
    assert fast_cleaner.process(text).lm_text == expected


def _clean(text, **kw):
    return TextCleaner(TextCleanerConfig(check_ner_process=False, **kw)).process(text).lm_text


# --- letters of the CJK punctuation block (々 〆 〇) are domain letters ---
@pytest.mark.parametrize("mode", ["placeholder", "reversible"])
def test_email_domain_with_iteration_mark(mode):
    assert "佐々木" not in _clean("Mail info@佐々木.jp now", replacement_mode=mode)


def test_email_domain_still_stops_at_cjk_punctuation():
    assert constants.EMAIL_REGEX.search("a@b.com（担当：鈴木）").group() == "a@b.com"
    assert constants.EMAIL_REGEX.search("a@b.com、次").group() == "a@b.com"


# --- full-width contact punctuation folds beside letters of any script ---
@pytest.mark.parametrize("text, tag", [
    ("taro＠例え.jp", "<EMAIL>"),
    ("山田＠example.com", "<EMAIL>"),
    ("taro＠例え．jp", "<EMAIL>"),
    ("https：／／例え.jp/パス", "<URL>"),
])
def test_fullwidth_contact_beside_cjk(text, tag):
    assert _clean(text) == tag


# --- ZWNJ/ZWJ inside an IDNA domain label ---
@pytest.mark.parametrize("fold", [True, False])
@pytest.mark.parametrize("text", ["info@نامه‌ای.ir", "info@क्‍ष.com"])
def test_email_domain_with_join_control(text, fold):
    assert _clean(text, fold_styled_characters=fold) == "<EMAIL>"


# --- full-width separators inside an address fold beside any script ---
@pytest.mark.parametrize("fold", [True, False])
@pytest.mark.parametrize("text", ["john@例え．テスト", "john@例え－テスト.jp", "josé＿smith@example.com",
                                  "山田＿太郎@example.com"])
def test_fullwidth_separator_in_address_beside_non_ascii(text, fold):
    assert _clean(text, fold_styled_characters=fold) == "<EMAIL>"


MODES = ["placeholder", "synthetic", "reversible"]


def _clean_training(text, mode="placeholder", **kw):
    cfg = TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode, **kw)
    return TextCleaner(cfg).process(text).lm_text


# --- combining marks in an email local part ---
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fold", [True, False])
@pytest.mark.parametrize("local", ["প্রিয়া", "பிரியா", "संपर्क"])
def test_email_local_part_with_combining_marks(local, fold, mode):
    out = _clean_training(f"mail {local}@example.com now", mode, fold_styled_characters=fold)
    assert local[:2] not in out and "example.com" not in out


@pytest.mark.parametrize("fold", [True, False])
def test_fullwidth_at_after_combining_mark(fold):
    assert _clean_training("प्रिया＠example.com", "placeholder", fold_styled_characters=fold) == "<EMAIL>"


# --- one circled local-part character before a full-width ＠ ---
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("email", ["ⓙ＠example.com", "ⓙ＠例え.jp"])
def test_single_circled_local_part_fullwidth_at(email, mode):
    out = _clean_training(f"mail {email} now", mode)
    assert "ⓙ" not in out and "＠" not in out and "例え" not in out


EMAILS = ["john@نامه\u200cای．ایران", "john@क्\u200dष．भारत", "مریم@نامه\u200cای．ایران"]


# --- a join control (ZWNJ/ZWJ) inside an address does not stop its full-width dot folding ---
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


# --- 1 (fuzz): a join control inside the TLD label, as in any other host label ---
@pytest.mark.parametrize("email", ["ab@يت.ط‌ة‌ق", "ab@नाम.भा‍रत", "ab@يت．ط‌ة‌ق"])
def test_join_control_in_tld_masked(email):
    out = TextCleaner(TextCleanerConfig(check_ner_process=False)).process(f"mail {email} now").lm_text
    assert out == "mail <EMAIL> now"


def _stream(text, **kw):
    stream = WindowedStreamingRedactor(TextCleaner(TextCleanerConfig(check_ner_process=False, **kw)),
                                       max_buffer_chars=48, overlap_chars=12)
    return "".join(stream.feed(text[i:i + 3]) for i in range(0, len(text), 3)) + stream.finish()


# --- IDNA dot separators (。 ｡) in a mail domain or URL host ---
CONTACTS = ["alice@acmemail。com", "alice@acmemail｡com", "https://www。acmemail。com/?token=SecretABC",
            "www。acmemail。com/?token=SecretABC"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("contact", CONTACTS)
def test_idna_dot_contact_masked(contact, mode):
    out = _clean_training(f"See {contact} now", mode)
    assert "acmemail" not in out and "Secret" not in out
    assert out.startswith("See ") and out.endswith(" now")


def test_idna_dot_contact_masked_in_stream():
    out = _stream("please write to alice@example。com today")
    assert "alice" not in out and "example" not in out


# --- squared Latin letters fold like circled ones ---
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("email", ["🄹🄾🄷🄽@acmemail.com", "JOHN@🄰🄲🄼🄴🄼🄰🄸🄻.com"])
def test_squared_letter_email_masked(email, mode):
    out = _clean_training(f"Mail {email} now", mode)
    assert "JOHN" not in out and "🄹" not in out and "acmemail" not in out and "🄰" not in out


# --- layout hints inside a non-Latin mail domain ---
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("hint", ["​", "­", "⁠"])
def test_layout_hint_in_idn_domain_masked(hint, mode):
    out = _clean_training(f"Mail info@例{hint}え.jp now", mode)
    assert "info" not in out and "例" not in out


def test_layout_hint_in_idn_domain_masked_in_stream():
    out = _stream("please write to info@例​え.jp today")
    assert "info" not in out and "例" not in out


def test_join_controls_in_domain_kept():
    assert fold_styled("john@نامه‌ای.ایران") == "john@نامه‌ای.ایران"


@pytest.mark.parametrize("email", ["alice@acmemail｡​com", "info@例え.テ​スト"])
def test_idna_dot_or_later_label_hint_masked(email):
    out = _clean_training(f"Mail {email} now")
    assert "alice" not in out and "info" not in out


def _cleaner(mode="placeholder", **kw):
    return TextCleaner(TextCleanerConfig.from_preset("training", check_ner_process=False, replacement_mode=mode, **kw))


def _stream_safe(cleaner, text, secret):
    for pad in range(0, 30):
        stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=80, overlap_chars=16)  # holds the URL
        out = stream.feed("x " * pad + text) + stream.finish()
        assert secret not in out, (pad, out)


# --- the IDNA middle dot between two l's (RFC 5892 A.3) stays inside a label ---
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


# --- a fold run takes every local-part character the email grammar takes, typeset dashes too ---
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


def test_entity_encoded_email_in_markup_is_replaced():
    text = "<p>Mail john&#64;example.com now</p>"
    out = TextCleaner(cfg=TextCleanerConfig(check_ner_process=False)).process(text).lm_text
    assert "john" not in out and "<EMAIL>" in out


MIDDLE_DOT_MAILS = ["Mail info@col·legi。cat now", "Mail info@col·legi｡cat now", "Mail josé＿col·legi@example.cat now"]


# with fold_styled_characters off an ideographic full stop is prose (as without the middle dot); ＿ still folds
@pytest.mark.parametrize("mode", ["placeholder", "synthetic", "reversible"])
@pytest.mark.parametrize("text,fold", [(t, True) for t in MIDDLE_DOT_MAILS] + [(MIDDLE_DOT_MAILS[2], False)])
def test_middle_dot_with_styled_separator_masked(text, mode, fold):
    out = _cleaner(mode, fold_styled_characters=fold).process(text).lm_text
    assert "legi" not in out and "josé" not in out and "info" not in out, out
    assert out.startswith("Mail ") and out.endswith(" now"), out


def test_middle_dot_json_stream_pickle_batch():
    cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False))
    for text in MIDDLE_DOT_MAILS:
        assert cleaner.process(text).lm_text == "Mail <EMAIL> now", text
        assert "legi" not in str(process_json(cleaner, {"note": text}).value)
        for pad in range(0, 30):
            stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=40, overlap_chars=16)
            out = stream.feed("x " * pad + text) + stream.finish()
            assert "legi" not in out, (pad, out)
    cfg = pickle.loads(pickle.dumps(cleaner.cfg))  # noqa: S301  # our own config, round-tripped
    out = TextCleaner(cfg).process_batch(MIDDLE_DOT_MAILS)
    assert [r.lm_text for r in out] == ["Mail <EMAIL> now"] * len(MIDDLE_DOT_MAILS)


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


@pytest.mark.parametrize("text, email", [
    ("お問い合わせはseikyu@example.com（担当：鈴木花子）まで。", "seikyu@example.com"),
    ("联系 a@b.cn。谢谢", "a@b.cn"),
    ("write a@b.com“quoted”", "a@b.com"),
    ("mail info@例え.jp now", "info@例え.jp"),
    ("user@उदाहरण.भारत ok", "user@उदाहरण.भारत"),
])
def test_email_domain_stops_at_cjk_and_fullwidth_punctuation(text, email):
    assert [m.group() for m in constants.EMAIL_REGEX.finditer(text)][0].endswith(email)


def test_email_keeps_the_name_after_fullwidth_bracket_visible_to_ner():
    cleaner = TextCleaner(cfg=TextCleanerConfig(check_ner_process=False))
    assert cleaner.process("seikyu@example.com（担当：鈴木花子）").lm_text == "<EMAIL>（担当：鈴木花子）"


@pytest.mark.parametrize("text, secret", [
    # small-form contact punctuation (﹫ ﹒ ﹕ ﹖ ﹦ ﹣ ﹢) folds like the full-width forms
    ("mail mx1eaq﹫zx9wnzm﹒de now", "mx1eaq"),
    ("see http﹕//悛搖鯰卆哯﹒net/9jmz﹖user﹦zt0mkdjuj8c now", "zt0mkdjuj8c"),
    ("host 214﹒44﹒189﹒208 now", "189"),
    ("call 018﹣2896 ext now", "2896"),
    # font plus (﬩, decomposition <font> +) folds as a phone prefix and email separator
    ("call ﬩91 21774 41978 now", "21774"),
    ("mail षीबनझ﬩iax4n@k6bb.net now", "षीबनझ"),
])
def test_small_and_font_contact_punctuation_folds(text, secret):
    assert secret not in _clean_training(text)


def test_full_stop_after_a_word_ending_in_a_mark_folds():
    # ै is a combining mark: the word before ． still ends there
    assert "g0up" not in _clean_training("see https://ळौछनै．de/g0up?token=w3gyy now")


def test_zero_width_joiner_inside_email_local_part():
    assert "घप" not in _clean_training("mail प्‍ॄघप@czgc0.org now")


# Prose of a script without spaces (Thai, CJK) glued to a Latin address ends it, so the address
# does not swallow the next one, and the next one starts right after the prose.
@pytest.mark.parametrize("text, secret", [
    # the first email's TLD took the Thai prose and the next local part's first letters
    ("ดูa@b.jpกขคcj9rf@xyz.nl ok", "9rf"),
    ("ดูa@b.ｊｐกขคcｊ9𝓇f﹫3t𝓍𝓃𝒋﹒ｎl ok", "3txnj"),
    # an email local part took the phone's last group and the Thai prose before it
    ("ฅฆ﹢𝟗𝟏 𝟒𝟐𝟒𝟎𝟐 𝟒𝟓𝟗𝟑𝟑หปตณ＋rxi੨＠byh੯ak．io", "42402"),
    # a URL query took the Thai prose and the phone's first group
    ("https://zem3d.com/f?id=a&sid=xyzปถว06 10 29 37 39ฝน", "10293739"),
    ("ｈｔｔｐｓ://ｚｅｍ３ｄ﹒ｃｏｍ/ｆ?ｓｉｄ=ｘｙｚปถว🯰🯶 🯱🯰 🯲🯹 🯳🯷 🯳🯹ฝน", "10293739"),
])
def test_spaceless_prose_glued_to_address_ends_it(text, secret):
    assert secret not in _clean_training(text).replace(" ", "")


@pytest.mark.parametrize("text, kept", [
    ("ติดต่อ cj9rf@xyz.nlขอบคุณ", "ขอบคุณ"),
    ("ดูhttps://x.comต่อ", "ต่อ"),
    ("詳細はhttps://example.jp/a/bまで", "まで"),
    ("メールtaro@example.jpまで", "メール"),
])
def test_glued_prose_stays_outside_the_address(text, kept):
    out = _clean_training(text)
    assert kept in out and ("<EMAIL>" in out or "<URL>" in out)


@pytest.mark.parametrize("text", [
    "https://th.wikipedia.org/wiki/ภาษาไทย",
    "https://example.jp/検索?q=東京",
    "info@佐々木.jp",
    "สมชาย@ตัวอย่าง.ไทย",
    "ณกฌห@мчз.jp",
    # a percent escape is one character, so a Thai/CJK domain after %40 stays in the query
    "https://x.in/o9v?id=zf0j%40ถมยฌฒ.in&sid=ei9k",
    "https://оиуф.net/m7w?q=3acu%㊵虘圜竿趴.jp&sid=iae5",
    "https://x.com/d5?user=axy෦q%෪0碵憐稕.de&sid=djt0lebzerj",
])
def test_addresses_in_one_script_stay_whole(text):
    assert _clean_training(f"ok {text} ok") in ("ok <URL> ok", "ok <EMAIL> ok")


# An invisible line-break hint (Cf: ZWSP, word joiner, soft hyphen) inside a value hid it from every regex.
@pytest.mark.parametrize("text, secret", [
    ("call 0671­427605 now", "0671"),
    ("call 06​ 71 42 76 05 now", "714276"),
    ("host 192.168​.1.1 ok", "192.168"),
    ("mail jan.de​vries@example.com ok", "jan.de"),
    ("mail jan@ex⁠ample.com ok", "jan@"),
    ("mail jan@例え​．jp ok", "jan@"),
    ("see https://exa­mple.com/a?id=12345 ok", "exa"),
])
def test_invisible_hint_inside_value_folds(text, secret):
    assert secret not in _cleaner().process(text).lm_text.replace(" ", "")


def test_invisible_hint_stays_with_fold_off():
    assert "­" in _cleaner(fold_styled_characters=False).process("call 0671­427605 now").lm_text
