"""Codex 0.8.0 full review, round 6: ids 1, 2, 3."""

import asyncio
import threading
import time

import pytest

from sct import TextCleaner
from sct.config import TextCleanerConfig
from sct.utils import constants
from tests.helpers import FakeRunDetector, WordCountTokenizer, word_ner


# --- 1: URL user info never runs across "/", "?" or markup to a later "@" ---

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


# --- 2: a route without the primary model is not split by the primary's count ---

def test_route_without_primary_counts_with_its_own_tokenizer():
    ner = word_ner(0, FakeRunDetector(min_words=2), budget=12)
    heavy = WordCountTokenizer(2)
    ner.tokenizer = heavy  # primary (ENGLISH), 2 tokens per word
    ner._ensemble_models = {"ENGLISH": ("ENGLISH",), "HINDI": ("HINDI",)}
    ner._tokenizers = {"ENGLISH": heavy, "HINDI": WordCountTokenizer(1)}
    text = "a b c d e John Smith f g h"  # 10 HINDI tokens, inside the budget of 12
    keys = ner._get_ensemble_keys("HINDI")
    assert [w[0] for w in ner._plan_windows(text, keys)] == [text]
    kw = {"positional_tags": ("PER",), "ner_confidence_threshold": 0.5}
    masked = text.replace("John Smith", "<PERSON>")
    assert ner.ner_process_detailed(text, language="HINDI", **kw).text == masked
    assert [r.text for r in ner.ner_process_batch_detailed([text], ["HINDI"], **kw)] == [masked]
    # A route that does use the primary still counts with it.
    assert len(ner._plan_windows(text, ner._get_ensemble_keys("ENGLISH"))) == 2


# --- 3: concurrent first MCP calls build one cleaner ---

def test_concurrent_first_calls_build_one_cleaner(monkeypatch):
    pytest.importorskip("fastmcp")
    from fastmcp import Client

    import sct.mcp_server as mcp_module

    built = []
    real = TextCleaner(cfg=TextCleanerConfig(
        check_ner_process=False, check_detect_language=False,
        check_statistical_model_processing=False))

    def slow_cleaner(cfg, **_):
        built.append(threading.get_ident())
        time.sleep(0.3)
        return real

    monkeypatch.setattr(mcp_module, "TextCleaner", slow_cleaner)
    server = mcp_module.create_mcp_server()

    async def run():
        async with Client(server) as client:
            return await asyncio.gather(*[
                client.call_tool("clean_text", {"text": "hi"}) for _ in range(4)])

    results = asyncio.run(run())
    assert len(built) == 1
    assert all(r.structured_content["text"] == "hi" for r in results)
