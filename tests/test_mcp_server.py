import json
import sys

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.mcp_server import (
    ToolError,
    create_mcp_server,
    sanitize_json,
    sanitize_text,
    summarize,
)
from sct.privacy import DetectedEntity, EntityFinding
from sct.utils.process_result import ProcessResult
from tests.helpers import client_call

SIGNAL_KEYS = [
    "language", "language_confidence", "language_source", "chars",
    "pii_count", "pii_chars", "pii_share", "pii_types", "regex_counts",
    "entity_counts", "sensitive", "warnings", "models",
]


def _cleaner(**overrides):
    return TextCleaner(cfg=TextCleanerConfig(
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
        **overrides,
    ))


class FakeCleaner:
    """Turns "Alice" into a PERSON finding and emails into EMAIL findings."""

    cfg = TextCleanerConfig(check_ner_process=False, json_key_cleaning="full")  # keys go through process()

    def process(self, text):
        if text not in ("Alice", "alice@example.com"):
            return ProcessResult(text, None, "ENGLISH", metadata={"source_chars": len(text)})
        label = "PERSON" if text == "Alice" else "EMAIL"
        return ProcessResult(
            f"<{label}>",
            "stat " + text,
            "ENGLISH",
            metadata={
                "source_chars": len(text),
                "entity_counts": {label: {"count": 1, "chars": len(text)}},
                "language_confidence": 0.9,
                "language_source": "detected",
                "provenance": {"backend": "onnx", "models": [
                    {"key": "ENGLISH", "id": "m", "revision": "r", "provider": "CPU"},
                ]},
                "findings": (EntityFinding(
                    DetectedEntity(label, 0.9, text, 0, len(text)),
                    "placeholder",
                    "default",
                ),),
            },
        )


def test_text_output_shape_and_order():
    result = sanitize_text(_cleaner(), "Email maria@example.com")

    assert list(result) == ["warnings", "summary", "text", "signals"]
    assert result["text"] == "Email <EMAIL>"
    assert list(result["signals"]) == SIGNAL_KEYS
    assert result["signals"]["pii_types"] == {"EMAIL": 1}
    assert result["warnings"] == result["signals"]["warnings"]
    assert "stat_text" not in json.dumps(result)
    assert result["summary"].startswith("1 PII item (EMAIL), 74% of text")


def test_full_detail_adds_value_free_findings():
    result = sanitize_text(FakeCleaner(), "Alice", detail="full")

    assert list(result) == ["warnings", "summary", "text", "signals", "findings"]
    assert result["findings"] == [{
        "entity_type": "PERSON", "score": 0.9, "start": 0, "end": 5,
        "action": "placeholder", "reason": "default",
    }]
    assert "Alice" not in json.dumps(result)
    assert result["signals"]["models"] == [
        {"key": "ENGLISH", "id": "m", "revision": "r", "provider": "CPU"},
    ]


def test_real_cleaner_output_has_no_raw_pii():
    text = "Mail maria@example.com or call +34 612 345 678 from 10.0.0.1"
    serialized = json.dumps(sanitize_text(_cleaner(), text, detail="full"))
    for raw in ("maria@example.com", "612 345 678", "10.0.0.1"):
        assert raw not in serialized


def test_json_keeps_key_path_rule_and_leaks_nothing():
    result = sanitize_json(
        FakeCleaner(), {"team": {"Alice": "alice@example.com"}}, detail="full",
    )
    serialized = json.dumps(result)

    assert list(result) == ["warnings", "summary", "value", "signals", "findings"]
    assert result["value"] == {"team": {"<PERSON>": "<EMAIL>"}}
    assert [item["path"] for item in result["findings"]] == [
        ["team", "@key", 0],
        ["team", "<PERSON>"],
    ]
    assert result["signals"]["pii_types"] == {"EMAIL": 1, "PERSON": 1}
    assert result["signals"]["chars"] == len("team") + len("Alice") + len("alice@example.com")
    assert len(result["signals"]["models"]) == 1
    assert "Alice" not in serialized
    assert "alice@example.com" not in serialized


def test_json_concise_and_empty_value_keep_the_same_signal_keys():
    result = sanitize_json(_cleaner(), {})

    assert list(result) == ["warnings", "summary", "value", "signals"]
    assert list(result["signals"]) == SIGNAL_KEYS
    assert "empty_input" in result["warnings"]
    assert result["summary"] == "No PII found, language unknown"


def test_json_language_is_char_weighted():
    class Mixed:
        cfg = TextCleanerConfig(check_ner_process=False, json_key_cleaning="full")

        def process(self, text):
            language = "HINDI" if len(text) > 3 else "ENGLISH"
            return ProcessResult(text, None, language, metadata={
                "source_chars": len(text),
                "language_confidence": 0.5 if len(text) > 5 else 1.0,
                "language_source": "detected",
            })

    signals = sanitize_json(Mixed(), ["abcd", "abcdefgh", "ab"])["signals"]

    assert signals["language"] == "HINDI"
    assert signals["language_confidence"] == pytest.approx((4 * 1.0 + 8 * 0.5) / 12)


def test_json_duplicate_key_is_a_tool_error():
    with pytest.raises(ToolError, match="unique"):
        sanitize_json(FakeCleaner(), {"Alice": 1, "<PERSON>": 2})


def test_summary_sentence():
    assert summarize({
        "pii_count": 2, "pii_types": {"EMAIL": 1, "PERSON": 1}, "pii_share": 0.04,
        "language": "HINDI", "language_confidence": 0.93,
    }) == "2 PII items (EMAIL, PERSON), 4% of text, language HINDI 0.93"
    assert summarize({
        "pii_count": 0, "pii_types": {}, "pii_share": 0.0,
        "language": "DUTCH", "language_confidence": None,
    }) == "No PII found, language DUTCH"


def test_mcp_server_reports_missing_optional_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastmcp", None)

    with pytest.raises(ImportError, match=r"squeakycleantext\[mcp\]"):
        create_mcp_server()


def test_bad_max_bytes_rejected():
    pytest.importorskip("fastmcp")
    with pytest.raises(ValueError, match="SCT_MCP_MAX_BYTES"):
        create_mcp_server(_cleaner(), max_bytes=-1)


def test_tools_declare_output_schema_and_annotations():
    pytest.importorskip("fastmcp")

    async def list_tools(client):
        return await client.list_tools()

    tools = {t.name: t for t in client_call(create_mcp_server(_cleaner()), list_tools)}

    assert set(tools) == {"clean_text", "clean_json"}
    for tool in tools.values():
        notes = tool.annotations
        assert notes.title
        assert notes.read_only_hint is True
        assert notes.destructive_hint is False
        assert notes.idempotent_hint is True
        assert notes.open_world_hint is True  # a cold cache downloads models
        schema = tool.output_schema
        assert set(schema["required"]) >= {"warnings", "summary", "signals"}
        assert "findings" in schema["properties"]
    assert tools["clean_text"].input_schema["properties"]["detail"]["enum"] == ["concise", "full"]


def test_tool_calls_return_structured_content_and_tool_errors():
    pytest.importorskip("fastmcp")
    from fastmcp.exceptions import ToolError as FastToolError

    server = create_mcp_server(_cleaner())

    async def calls(client):
        text = await client.call_tool("clean_text", {"text": "Email maria@example.com"})
        data = await client.call_tool("clean_json", {"value": {"email": "maria@example.com"}})
        return text.structured_content, data.structured_content

    text, data = client_call(server, calls)
    assert list(text) == ["warnings", "summary", "text", "signals"]
    assert text["text"] == "Email <EMAIL>"
    assert data["value"] == {"email": "<EMAIL>"}

    async def pii(client):
        return await client.call_tool("clean_text", {"text": "Maria", "pii": True})

    with pytest.raises(FastToolError, match="not configured for PII mode"):
        client_call(server, pii)


def test_preset_picks_the_built_in_cleaner_config(monkeypatch):
    pytest.importorskip("fastmcp")
    import sct.mcp_server as mcp_module

    with pytest.raises(ValueError, match="unknown preset"):
        create_mcp_server(preset="nope")
    seen = []
    monkeypatch.setattr(mcp_module, "TextCleaner", lambda cfg, **_: seen.append(cfg) or _cleaner())
    monkeypatch.setenv("SCT_MCP_PRESET", "training")
    client_call(create_mcp_server(), lambda c: c.call_tool("clean_text", {"text": "x"}))
    assert seen[0].check_to_ascii_unicode is False
