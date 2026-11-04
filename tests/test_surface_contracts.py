"""Regression tests for adapter, MCP, CLI, score and token-map contracts."""

import json
import os
from io import StringIO
from types import SimpleNamespace

import pytest

import sct
from sct import AnonymizationMap, TextCleaner, TextCleanerConfig
from sct.adapters import RAGChunk, is_sensitive_result, process_dataframe, process_rag_chunks
from sct.cli import main as clean_main
from sct.mcp_server import ToolError, read_under_root, sanitize_json, sanitize_text
from sct.report_cli import main as report_main
from sct.score import SCORE_A_COLUMNS, score_records
from sct.signals import PUBLIC_VIEW_KEYS, public_view


def _regex_cleaner(**overrides):
    return TextCleaner(cfg=TextCleanerConfig(
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
        **overrides,
    ))


def _fake_result(**metadata):
    return SimpleNamespace(language=None, metadata=metadata)


def test_one_sensitivity_rule():
    only_findings = _fake_result(findings=["x"])
    assert is_sensitive_result(only_findings) is True
    assert public_view(only_findings)["sensitive"] is True
    assert set(public_view(_fake_result())) == set(PUBLIC_VIEW_KEYS)


def test_rag_chunk_reports_metadata_pii():
    chunk = RAGChunk(id="1", text="No contacts here", metadata={"owner": "maria@example.com", "page": 1})

    result = process_rag_chunks(_regex_cleaner(), [chunk])[0]

    assert result.sensitive is True and result.metadata["sensitive"] is True
    assert result.signals["pii_types"] == {"EMAIL": 1}
    # metadata["sct"] keeps describing the text alone.
    assert result.metadata["sct"]["pii_count"] == 0


def test_rag_chunk_without_metadata_text_has_the_text_view():
    result = process_rag_chunks(_regex_cleaner(), [RAGChunk(id="1", text="Plain words", metadata={})])[0]

    assert result.signals == result.metadata["sct"]
    assert "empty_input" not in result.signals["warnings"]


def test_process_dataframe_reports_sensitivity_and_takes_one_column_name():
    pd = pytest.importorskip("pandas")
    # "t" must stay untouched: a plain string used to be read as its characters.
    frame = pd.DataFrame({"text": ["a@example.com"], "t": ["b@example.com"]})

    result = process_dataframe(_regex_cleaner(), frame, columns="text")

    assert result.frame.values.tolist() == [["<EMAIL>", "b@example.com"]]
    assert result.sensitive is True and result.signals["pii_count"] == 1
    assert process_dataframe(_regex_cleaner(), frame[["t"]].assign(t=["plain"]), columns="t").sensitive is False


def test_mcp_full_warns_when_counted_entities_have_no_findings():
    cleaner = _regex_cleaner()
    process = cleaner.process

    def with_entity(text):
        result = process(text)
        result.metadata["entity_counts"] = {"PERSON": {"count": 1, "chars": 5}}
        return result

    cleaner.process = with_entity

    assert "findings_not_collected" in sanitize_text(cleaner, "Maria", "full")["warnings"]
    assert "findings_not_collected" not in sanitize_text(cleaner, "Maria")["warnings"]
    assert "findings_not_collected" in sanitize_json(cleaner, ["Maria"], "full")["warnings"]
    assert "findings_not_collected" not in sanitize_text(_regex_cleaner(), "a@example.com", "full")["warnings"]


def test_sanitize_json_gives_the_key_remedy_only_for_duplicate_keys():
    with pytest.raises(ToolError, match="stay unique"):
        sanitize_json(_regex_cleaner(json_key_cleaning="full"), {"b": 1, "c": 2})
    reversible = _regex_cleaner(replacement_mode="reversible")
    with pytest.raises(ToolError, match="reversible mode") as caught:
        sanitize_json(reversible, {"k": "v"})
    assert "stay unique" not in str(caught.value)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="path input needs O_NOFOLLOW")
def test_missing_root_error_does_not_name_the_folder(tmp_path):
    root = tmp_path / "secret-folder-name"

    with pytest.raises(ToolError) as caught:
        read_under_root(str(root), "a.txt")

    assert "secret-folder-name" not in str(caught.value)


def test_cli_reports_unreadable_input_as_usage_error(tmp_path):
    with pytest.raises(SystemExit) as missing:
        clean_main(["--no-ner", str(tmp_path / "absent.txt")])
    binary = tmp_path / "binary.txt"
    binary.write_bytes(b"\xff\xfe")
    with pytest.raises(SystemExit) as undecodable:
        clean_main(["--no-ner", str(binary)])

    assert missing.value.code == undecodable.value.code == 2


def test_cli_writes_no_masked_text_when_token_map_fails(tmp_path):
    target = tmp_path / "target.json"
    link = tmp_path / "tokens.json"
    link.symlink_to(target)
    output = StringIO()

    with pytest.raises(SystemExit) as caught:
        clean_main(
            ["--no-ner", "--replacement-mode", "reversible", "--token-map", str(link)],
            stdin=StringIO("Email maria@example.com"),
            stdout=output,
        )

    assert caught.value.code == 2 and output.getvalue() == "" and not target.exists()


def test_cli_refuses_reversible_mode_without_fchmod_before_cleaning(tmp_path, monkeypatch):
    monkeypatch.delattr(os, "fchmod", raising=False)
    output = StringIO()

    with pytest.raises(SystemExit) as caught:
        clean_main(
            ["--no-ner", "--replacement-mode", "reversible", "--token-map", str(tmp_path / "t.json")],
            stdin=StringIO("Email maria@example.com"),
            stdout=output,
        )

    assert caught.value.code == 2 and output.getvalue() == ""


def test_cli_reports_model_load_error_as_usage_error(monkeypatch):
    def fail(self, text):
        raise sct.ModelLoadError("Model loading failed")

    monkeypatch.setattr(TextCleaner, "process", fail)

    with pytest.raises(SystemExit) as caught:
        clean_main(["--no-ner"], stdin=StringIO("x"), stdout=StringIO())

    assert caught.value.code == 2


def test_report_names_the_malformed_jsonl_line(tmp_path, capsys):
    source = tmp_path / "rows.jsonl"
    source.write_text('{"text": "ok"}\n{broken\n', encoding="utf-8")

    with pytest.raises(SystemExit) as caught:
        report_main([str(source), "--no-ner", "--out", str(tmp_path / "out.json")])

    assert caught.value.code == 2 and "line 2" in capsys.readouterr().err


def test_report_reports_missing_input_as_usage_error(tmp_path):
    with pytest.raises(SystemExit) as caught:
        report_main([str(tmp_path / "absent.jsonl"), "--out", str(tmp_path / "out.json")])

    assert caught.value.code == 2


def test_score_records_scorer_alone_turns_quality_on():
    result = SimpleNamespace(signals=_regex_cleaner().process("a").signals, lm_text="a")

    row = score_records([result], ["a"], scorer=lambda texts: [0.5 for _ in texts])[0]

    assert row["sct_quality"] == 0.5 and row["sct_minhash"] is None
    assert tuple(row)[:len(SCORE_A_COLUMNS)] == SCORE_A_COLUMNS


def test_anon_map_round_trip_keeps_allocating_new_placeholders():
    original = AnonymizationMap(reserved={"<PERSON_1>"})
    original.add(original.next_placeholder("PERSON"), "Maria", "PERSON", 0, 5)

    restored = AnonymizationMap.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored.next_placeholder("PERSON") == "<PERSON_2>"
    assert restored.reserved == {"<PERSON_1>"}


def test_anon_map_loads_payload_without_reserved():
    payload = {"session_id": "s", "entries": [
        {"placeholder": "<EMAIL_3>", "original": "a@example.com", "entity_type": "EMAIL", "start": 0, "end": 13},
    ]}

    assert AnonymizationMap.from_dict(payload).next_placeholder("EMAIL") == "<EMAIL_4>"
