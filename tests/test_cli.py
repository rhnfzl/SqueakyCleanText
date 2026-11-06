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
