import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.adapters import (
    RAGChunk,
    process_dataframe,
    process_json,
    process_rag_chunks,
)


def _regex_cleaner():
    return TextCleaner(cfg=TextCleanerConfig(
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
    ))


def test_process_json_preserves_structure_and_non_text_values():
    source = {
        "user": {
            "email": "maria@example.com",
            "age": 42,
        },
        "messages": ["Call +1-202-555-0147", None],
    }

    result = process_json(_regex_cleaner(), source)

    assert result.value == {
        "user": {"email": "<EMAIL>", "age": 42},
        "messages": ["Call <PHONE>", None],
    }
    assert source["user"]["email"] == "maria@example.com"


def test_process_json_cleans_string_keys():
    result = process_json(
        _regex_cleaner(),
        {"maria@example.com": "owner"},
    )

    assert result.value == {"<EMAIL>": "owner"}


def test_structured_adapters_reject_reversible_mode_without_map_contract():
    cleaner = TextCleaner(cfg=TextCleanerConfig(
        replacement_mode="reversible",
        check_ner_process=False,
    ))

    with pytest.raises(ValueError, match="do not support reversible mode"):
        process_json(cleaner, {"email": "maria@example.com"})


def test_process_rag_chunks_adds_sensitivity_metadata():
    chunks = [
        RAGChunk(id="one", text="Email maria@example.com", metadata={"source": "a"}),
        RAGChunk(id="two", text="No contacts here", metadata={"source": "b"}),
        RAGChunk(id="three", text="Café", metadata={"source": "c"}),
    ]

    results = process_rag_chunks(_regex_cleaner(), chunks)

    assert results[0].text == "Email <EMAIL>"
    assert results[0].metadata["sensitive"] is True
    assert results[0].metadata["source"] == "a"
    assert results[1].metadata["sensitive"] is False
    assert results[2].metadata["sensitive"] is False


def test_process_dataframe_copies_and_processes_selected_columns():
    class AtIndexer:
        def __init__(self, frame):
            self.frame = frame

        def __getitem__(self, key):
            row, column = key
            return self.frame.data[row][column]

        def __setitem__(self, key, value):
            row, column = key
            self.frame.data[row][column] = value

    class FakeFrame:
        def __init__(self, data):
            self.data = {row: dict(values) for row, values in data.items()}
            self.index = list(data)
            self.columns = list(next(iter(data.values())))
            self.at = AtIndexer(self)

        def copy(self, deep=True):
            return FakeFrame(self.data)

    source = FakeFrame({
        0: {"body": "Email a@example.com", "label": "keep@example.com"},
    })

    result = process_dataframe(_regex_cleaner(), source, columns=("body",))

    assert result.frame.at[0, "body"] == "Email <EMAIL>"
    assert result.frame.at[0, "label"] == "keep@example.com"
    assert source.at[0, "body"] == "Email a@example.com"


def test_sensitive_counts_markers_already_in_source():
    from sct.adapters import is_sensitive_result

    cleaner = TextCleaner(cfg=TextCleanerConfig(
        check_replace_html=False,
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
    ))
    source = "<EMAIL> alice@example.com"

    result = cleaner.process(source)

    assert result.lm_text == "<EMAIL> <EMAIL>"
    assert is_sensitive_result(result) is True
    assert is_sensitive_result(cleaner.process("<EMAIL> only")) is False


def test_sensitive_from_entity_counts_not_markers():
    from types import SimpleNamespace

    from sct.adapters import is_sensitive_result

    # NER output as observed with ONNX models; no model download here.
    found = SimpleNamespace(
        lm_text="<PERSON>",
        metadata={"regex_counts": {}, "entity_counts": {"PERSON": {"count": 1, "chars": 5}}},
    )
    assert is_sensitive_result(found) is True
    assert is_sensitive_result(SimpleNamespace(lm_text="<PERSON>", metadata=None)) is False


def test_process_json_keys_get_pii_steps_only_by_default():
    # The library default drops isolated letters and ASCII-folds; that used to
    # turn {"b": 1, "c": 2} into two "" keys (duplicate-key error) and fold keys.
    source = {"b": 1, "c": 2, "München 日本語": "x", "to maria@example.com": "k"}

    result = process_json(_regex_cleaner(), source)

    assert list(result.value) == ["b", "c", "München 日本語", "to <EMAIL>"]
    assert result.signals["regex_counts"]["EMAIL"]["count"] == 1


def test_process_json_key_cleaning_full_and_off():
    def cleaner(mode):
        return TextCleaner(cfg=TextCleanerConfig(
            check_ner_process=False,
            check_detect_language=False,
            check_statistical_model_processing=False,
            json_key_cleaning=mode,
        ))

    with pytest.raises(ValueError, match="duplicate key"):
        process_json(cleaner("full"), {"b": 1, "c": 2})
    off = process_json(cleaner("off"), {"to maria@example.com": "k"})
    assert off.value == {"to maria@example.com": ""}
    with pytest.raises(ValueError, match="json_key_cleaning"):
        TextCleanerConfig(json_key_cleaning="raw")
