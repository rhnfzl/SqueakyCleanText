"""Fuzz round 3 (research/0.8.0-fuzz-round3.md): one regression per fixed root cause."""

import pytest

from sct import TextCleaner
from sct.presets import preset_config


def _clean(text, **overrides):
    return TextCleaner(cfg=preset_config("training", check_ner_process=False, **overrides)).process(text).lm_text


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
    assert secret not in _clean(text).replace(" ", "")


@pytest.mark.parametrize("text, kept", [
    ("ติดต่อ cj9rf@xyz.nlขอบคุณ", "ขอบคุณ"),
    ("ดูhttps://x.comต่อ", "ต่อ"),
    ("詳細はhttps://example.jp/a/bまで", "まで"),
    ("メールtaro@example.jpまで", "メール"),
])
def test_glued_prose_stays_outside_the_address(text, kept):
    out = _clean(text)
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
    assert _clean(f"ok {text} ok") in ("ok <URL> ok", "ok <EMAIL> ok")
