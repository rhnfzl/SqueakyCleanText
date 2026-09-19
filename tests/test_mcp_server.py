import sys
from types import SimpleNamespace

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.mcp_server import create_mcp_server, sanitize_text
from sct.privacy import DetectedEntity, EntityFinding
from sct.utils.process_result import ProcessResult


def test_mcp_handler_returns_structured_cleaning_result():
    cleaner = TextCleaner(cfg=TextCleanerConfig(
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
    ))

    result = sanitize_text(cleaner, "Email maria@example.com")

    assert result == {
        "text": "Email <EMAIL>",
        "stat_text": None,
        "language": "ENGLISH",
        "findings": [],
        "policy": None,
    }


def test_mcp_handler_does_not_return_source_entity_values():
    class FakeCleaner:
        def process(self, text):
            return ProcessResult(
                "<PERSON>",
                None,
                "ENGLISH",
                metadata={
                    "findings": (
                        EntityFinding(
                            DetectedEntity("PERSON", 0.9, text, 0, len(text)),
                            "placeholder",
                            "policy-rule",
                        ),
                    ),
                },
            )

    result = sanitize_text(FakeCleaner(), "Maria")

    assert "Maria" not in str(result["findings"])
    assert result["findings"][0]["entity"]["entity_type"] == "PERSON"


def test_mcp_server_registers_text_and_json_tools_without_optional_package(
    monkeypatch,
):
    class FakeFastMCP:
        def __init__(self, name, *, instructions):
            self.name = name
            self.instructions = instructions
            self.tools = {}

        def tool(self, function):
            self.tools[function.__name__] = function
            return function

    monkeypatch.setitem(
        sys.modules,
        "fastmcp",
        SimpleNamespace(FastMCP=FakeFastMCP),
    )
    cleaner = TextCleaner(cfg=TextCleanerConfig(
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
    ))

    server = create_mcp_server(cleaner)

    assert server.name == "SqueakyCleanText"
    assert server.tools["clean_text"]("Email maria@example.com")["text"] == (
        "Email <EMAIL>"
    )
    assert server.tools["clean_json"](
        {"email": "maria@example.com"}
    )["value"] == {"email": "<EMAIL>"}
    with pytest.raises(ValueError, match="not configured for PII"):
        server.tools["clean_text"]("Maria", pii=True)


def test_mcp_server_reports_missing_optional_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastmcp", None)

    with pytest.raises(ImportError, match=r"squeakycleantext\[mcp\]"):
        create_mcp_server()
