from io import StringIO
import json

import pytest

from sct.cli import main


def test_cli_processes_standard_input_without_loading_ner():
    output = StringIO()

    exit_code = main(
        ["--no-ner"],
        stdin=StringIO("Email maria@example.com"),
        stdout=output,
    )

    assert exit_code == 0
    assert output.getvalue() == "Email <EMAIL>\n"


def test_cli_reversible_mode_writes_restricted_token_map(tmp_path):
    output = StringIO()
    token_map = tmp_path / "tokens.json"
    token_map.write_text("{}")
    token_map.chmod(0o644)

    exit_code = main(
        [
            "--no-ner",
            "--replacement-mode",
            "reversible",
            "--token-map",
            str(token_map),
        ],
        stdin=StringIO("Email maria@example.com"),
        stdout=output,
    )

    assert exit_code == 0
    assert output.getvalue() == "Email <EMAIL_0>\n"
    assert json.loads(token_map.read_text())["entries"][0]["original"] == (
        "maria@example.com"
    )
    assert token_map.stat().st_mode & 0o777 == 0o600


def test_cli_processes_json_file_input(tmp_path):
    source = tmp_path / "input.json"
    source.write_text('{"email": "maria@example.com"}')
    output = StringIO()

    exit_code = main(
        [str(source), "--no-ner", "--json-input"],
        stdout=output,
    )

    assert exit_code == 0
    assert json.loads(output.getvalue()) == {"email": "<EMAIL>"}


def test_cli_reversible_mode_requires_token_map():
    with pytest.raises(SystemExit):
        main(
            ["--no-ner", "--replacement-mode", "reversible"],
            stdin=StringIO("Email maria@example.com"),
        )


def test_cli_rejects_reversible_json_input(tmp_path):
    with pytest.raises(SystemExit):
        main([
            "--no-ner",
            "--json-input",
            "--replacement-mode",
            "reversible",
            "--token-map",
            str(tmp_path / "tokens.json"),
        ])


def test_cli_preset_training_keeps_non_latin_text_and_short_keys():
    output = StringIO()

    main(
        ["--no-ner", "--preset", "training", "--json-input"],
        stdin=StringIO('{"k": "日本語 maria@example.com München"}'),
        stdout=output,
    )

    assert json.loads(output.getvalue()) == {"k": "日本語 <EMAIL> München"}


# --- empty reversible input still carries a map, and the CLI writes it ---
def test_cli_reversible_empty_input_writes_empty_map(tmp_path):
    token_map = tmp_path / "tokens.json"
    exit_code = main(
        ["--no-ner", "--replacement-mode", "reversible", "--token-map", str(token_map)],
        stdin=StringIO(""), stdout=StringIO(),
    )
    assert exit_code == 0
    assert json.loads(token_map.read_text())["entries"] == []


def test_cli_reports_duplicate_json_key_as_usage_error(monkeypatch):
    import io
    from sct import cli
    monkeypatch.setattr(cli, "process_json", lambda *a: (_ for _ in ()).throw(
        ValueError("Cleaning JSON keys produced a duplicate key")))
    with pytest.raises(SystemExit) as exc:
        cli.main(["--json-input", "--no-ner"], stdin=io.StringIO('{"a": 1}'), stdout=io.StringIO())
    assert exc.value.code == 2


@pytest.mark.parametrize("processing", ["legacy", "unified"])
def test_cli_file_keeps_source_coordinates_until_cleaning(tmp_path, processing):
    from sct import TextCleaner
    from sct.presets import preset_config

    text = "Header\r\nEmail demo@example.test\r\nPhone +31 6 12345678\r\nIP 192.0.2.7\r\n"
    path = tmp_path / "input.txt"
    path.write_bytes(text.encode("utf-8"))
    expected = TextCleaner(cfg=preset_config(
        "training", check_ner_process=False, pii_processing=processing,
    )).process(text)

    output = StringIO()
    main([str(path), "--preset", "training", "--no-ner", "--processing", processing, "--signals"],
         stdout=output)
    payload = json.loads(output.getvalue())

    assert all(raw not in payload["text"] for raw in ("demo@example.test", "+31 6 12345678", "192.0.2.7"))
    assert all(tag in payload["text"] for tag in ("<EMAIL>", "<PHONE>", "<IP>"))
    assert payload["text"] == expected.lm_text
    assert payload["signals"]["chars"] == len(text)
    assert payload["signals"] == json.loads(json.dumps(expected.signals))


def test_cli_unified_ner_requires_explicit_network_permission(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--processing", "unified"], stdin=StringIO("Alex Example"))
    assert error.value.code == 2
    assert "error: Unified model detection requires --allow-network-detectors." in capsys.readouterr().err


@pytest.mark.needs_model
def test_cli_unified_ner_can_enable_network_detectors(monkeypatch):
    from dataclasses import replace

    from sct import TextCleaner, cli
    from tests.conftest import TEST_NER_MODELS, TEST_NER_REVISIONS

    def test_cleaner(cfg):
        return TextCleaner(replace(cfg, ner_models=TEST_NER_MODELS, ner_model_revisions=TEST_NER_REVISIONS))

    monkeypatch.setattr(cli, "TextCleaner", test_cleaner)
    output = StringIO()
    main(["--processing", "unified", "--allow-network-detectors", "--preset", "training"],
         stdin=StringIO("John Smith lives in London."), stdout=output)
    assert output.getvalue() == "<PERSON> lives in <LOCATION>.\n"
