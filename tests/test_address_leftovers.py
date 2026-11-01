"""0.8.0 fuzz leftovers: a long mail local part, IDNA CONTEXTO punctuation in a host, circled letters in an address."""

import time

import pytest

from sct import TextCleaner
from sct.presets import preset_config
from sct.utils import constants


def _cleaner(**kw):
    return TextCleaner(cfg=preset_config("training", check_ner_process=False, **kw))


# --- 1: a local part longer than 64 characters is masked whole, never left as text ---

@pytest.mark.parametrize("size", [64, 65, 200, 5000])
@pytest.mark.parametrize("fold", [True, False])
def test_long_local_part_masked(size, fold):
    out = _cleaner(fold_styled_characters=fold).process("mail " + "a" * size + "@example.com now").lm_text
    assert out == "mail <EMAIL> now"


@pytest.mark.parametrize("local", ["x!" + "a" * 70, "a-b." * 30 + "c", "ดู" + "a" * 70])
def test_long_local_part_with_punctuation_or_script_change(local):
    out = _cleaner().process(f"mail {local}@example.com now").lm_text
    assert "@" not in out and "a" * 10 not in out and "<EMAIL>" in out, out


@pytest.mark.parametrize("make", [lambda n: "a" * n + "@example.com", lambda n: "a!" * n, lambda n: "a" * n + " "])
def test_long_local_part_linear(make):
    def seconds(text):
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            constants.EMAIL_REGEX.search(text)
            best = min(best, time.perf_counter() - start)
        return best
    seconds(make(10))
    small, large = seconds(make(2000)), seconds(make(8000))
    assert large < 8 * small + 0.05, (small, large)


def test_short_local_part_after_punctuation_unchanged():
    cleaner = _cleaner()
    assert cleaner.process("say 'alice@example.com' now").lm_text == "say '<EMAIL>' now"
    assert cleaner.process("say !alice@example.com now").lm_text == "say !<EMAIL> now"


# --- 2: RFC 5892 CONTEXTO punctuation inside a host label ---

@pytest.mark.parametrize("fold", [True, False])
@pytest.mark.parametrize("text, plain", [
    ("https://例・え.jp/?token=SecretABC", "<URL>"),
    ("https://テ・スト.jp/?token=SecretABC", "<URL>"),
    ("mail a@例・え.jp ok", "mail <EMAIL> ok"),
    ("https://צ׳ה.co.il/?token=SecretABC", "<URL>"),
    ("mail a@צה״ל.co.il ok", "mail <EMAIL> ok"),
])
def test_contexto_punctuation_in_host(text, plain, fold):
    assert _cleaner(fold_styled_characters=fold).process(text).lm_text == plain


def test_contexto_punctuation_with_wide_query_folds():
    assert _cleaner().process("https://例・え.jp／？token=SecretABC").lm_text == "<URL>"


@pytest.mark.parametrize("regex, head", [(constants.EMAIL_REGEX, "info@"), (constants.URL_REGEX, "https://")])
def test_middle_dot_run_without_tld_is_linear(regex, head):
    # each dot between Han used to have two parses, so a host with no TLD backtracked 2^n ways
    start = time.perf_counter()
    regex.search(head + "例・" * 40 + "え")
    assert time.perf_counter() - start < 1.0


@pytest.mark.parametrize("text", ["see x・y.com ok", "リスト①・②です", "a@b.c・d", "見て・https://x.jp"])
def test_middle_dot_outside_kana_label_stays_punctuation(text):
    out = _cleaner().process(text).lm_text
    assert "・" in out, out


# --- 3: circled and squared letters inside an address run fold whatever sits beside them ---

@pytest.mark.parametrize("text", ["ab@éⓣ․net", "a@Ŗ🅂Ń.io", "ⓐé@例.jp", "see https://Ŗ🅂Ń.io/?token=S"])
def test_circled_letters_in_address_run_fold(text):
    out = _cleaner().process(text).lm_text
    assert "@" not in out and "token" not in out and ("<EMAIL>" in out or "<URL>" in out), out


@pytest.mark.parametrize("text", ["① 見て", "リスト①・②です", "Ⓐ plan, Ⓑ plan", "見てⓐ。"])
def test_circled_prose_outside_address_stays(text):
    assert _cleaner().process(text).lm_text == text
