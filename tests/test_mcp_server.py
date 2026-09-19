from sct import TextCleaner, TextCleanerConfig
from sct.mcp_server import sanitize_text
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
