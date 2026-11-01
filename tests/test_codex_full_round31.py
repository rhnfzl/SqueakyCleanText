"""Codex 0.8.0 full review, round 31: ids 1, 2 and 3."""

import copy
import dataclasses
import pickle

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.presets import preset_config
from sct.streaming import WindowedStreamingRedactor

FAST = preset_config("training", check_ner_process=False)
DEFAULT = TextCleanerConfig(check_ner_process=False)


# --- 1: a closing bracket followed by a URL sub-delimiter (; , $ ...) and more URL text stays in the URL ---

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


# --- 2: the stream guard decodes entities to a fixpoint, keeps multi-character decodes and repairs mojibake ---

@pytest.mark.parametrize("chunk", [1024, 4096, None])
@pytest.mark.parametrize("value, token", [
    ("06 12&amp;#32;34 56 78", "<PHONE>"),
    ("15 January&ThickSpace;1990", "<DATE>"),
    ("15 fÃ©vrier 1990", "<DATE>"),
])
def test_stream_keeps_repaired_token_whole(chunk, value, token):
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False, check_replace_dates=True))
    text = "Introduction. " * 300 + "Value " + value + " here. " + "Afterwards. " * 20
    stream = WindowedStreamingRedactor(cleaner)
    size = chunk or len(text)
    out = "".join(stream.feed(text[i:i + size]) for i in range(0, len(text), size)) + stream.finish()
    assert f"Value {token} here." in cleaner.process(text).lm_text
    assert f"Value {token} here." in out


# --- 3: a PII label map kept after replace() to another mode survives pickle, deepcopy and Haystack ---

def _kept_map_config():
    pii = TextCleanerConfig.from_preset("training", ner_mode="pii", language="ENGLISH")
    return dataclasses.replace(pii, ner_mode="standard")


def test_kept_pii_label_map_survives_round_trip():
    cfg = _kept_map_config()
    assert cfg.gliner_label_map["name"] == "PER"
    for loaded in (pickle.loads(pickle.dumps(cfg)), copy.deepcopy(cfg)):  # noqa: S301 - own bytes
        assert dict(loaded.gliner_label_map) == dict(cfg.gliner_label_map)
        assert loaded == cfg


def test_kept_pii_label_map_survives_haystack():
    pytest.importorskip("haystack")
    from sct.integrations.haystack import SCTDocumentCleaner

    cfg = _kept_map_config()
    loaded = SCTDocumentCleaner.from_dict(SCTDocumentCleaner(config=cfg).to_dict())._config
    assert dict(loaded.gliner_label_map) == dict(cfg.gliner_label_map)
