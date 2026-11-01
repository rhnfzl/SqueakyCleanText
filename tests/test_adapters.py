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
    pd = pytest.importorskip("pandas")
    source = pd.DataFrame({"body": ["Email a@example.com"], "label": ["keep@example.com"]})

    result = process_dataframe(_regex_cleaner(), source, columns=("body",))

    assert result.frame.at[0, "body"] == "Email <EMAIL>"
    assert result.frame.at[0, "label"] == "keep@example.com"
    assert source.at[0, "body"] == "Email a@example.com"


def test_process_dataframe_cleans_duplicate_row_and_column_labels():
    pd = pytest.importorskip("pandas")
    rows = pd.DataFrame({"text": ["a@example.com", "b@example.com"]}, index=[0, 0])
    cols = pd.DataFrame([["a@example.com", "b@example.com"]], columns=["text", "text"])

    assert list(process_dataframe(_regex_cleaner(), rows, columns=("text",)).frame["text"]) == [
        "<EMAIL>", "<EMAIL>"]
    assert process_dataframe(_regex_cleaner(), cols, columns=("text",)).frame.values.tolist() == [
        ["<EMAIL>", "<EMAIL>"]]


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
