"""0.8.0: the library default, sct-clean and sct-mcp keep every script; ASCII folding is opt-in."""

from io import StringIO

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.cli import main

TEXT = "München 日本語 Москва mail maria@example.com"
FOLDED = "Munchen Ri Ben Yu Moskva mail <EMAIL>"


def _cfg(**overrides):
    return TextCleanerConfig(check_ner_process=False, check_statistical_model_processing=False, **overrides)


def test_library_default_keeps_scripts_and_ascii_is_an_option():
    assert TextCleaner(cfg=_cfg()).process(TEXT).lm_text == "München 日本語 Москва mail <EMAIL>"
    assert TextCleaner(cfg=_cfg(check_to_ascii_unicode=True)).process(TEXT).lm_text == FOLDED


def test_cli_keeps_scripts_by_default_and_to_ascii_folds():
    for argv, expected in ((["--no-ner"], "München 日本語 Москва mail <EMAIL>"),
                           (["--no-ner", "--to-ascii"], FOLDED),
                           (["--no-ner", "--to-ascii", "--preset", "training"], FOLDED)):
        out = StringIO()
        assert main(argv, stdin=StringIO(TEXT), stdout=out) == 0
        assert " ".join(out.getvalue().split()) == expected  # training keeps unidecode's double space


def test_mcp_keeps_scripts_by_default_and_env_folds(monkeypatch):
    pytest.importorskip("fastmcp")
    import sct.mcp_server as mcp_module
    from tests.helpers import client_call

    seen = []
    monkeypatch.setattr(mcp_module, "TextCleaner", lambda cfg, **_: seen.append(cfg) or TextCleaner(cfg=_cfg()))
    call = lambda c: c.call_tool("clean_text", {"text": "x"})  # noqa: E731
    client_call(mcp_module.create_mcp_server(), call)
    monkeypatch.setenv("SCT_MCP_TO_ASCII", "1")
    client_call(mcp_module.create_mcp_server(), call)
    assert [cfg.check_to_ascii_unicode for cfg in seen] == [False, True]


@pytest.mark.parametrize("mode", ["placeholder", "reversible", "synthetic"])
def test_ipv4_in_other_digit_scripts_is_replaced(mode):
    # Without ASCII folding, "١٩٢.١٦٨.١.١" used to leak as "<NUMBER>.١.١".
    cleaner = TextCleaner(cfg=_cfg(replacement_mode=mode, check_replace_numbers=False, check_to_ascii_unicode=False))
    for text, ip in (("IP ١٩٢.١٦٨.١.١ ok", "١٩٢.١٦٨.١.١"), ("आईपी ८.८.८.८ है", "८.८.८.८")):
        result = cleaner.process(text)
        assert not any(part in result.lm_text for part in ip.split(".")), result.lm_text
        assert result.metadata["regex_counts"]["IP"]["count"] == 1


def test_non_ascii_digit_runs_that_are_not_ips_are_kept():
    cleaner = TextCleaner(cfg=_cfg(check_replace_numbers=False))
    assert cleaner.process("٩٩٩.١.١.١ ١.٢.٣.٤.٥").lm_text == "٩٩٩.١.١.١ ١.٢.٣.٤.٥"
    public = TextCleaner(cfg=_cfg(replace_ips_public_only=True))
    assert public.process("١٩٢.١٦٨.١.١ ٨.٨.٨.٨").lm_text == "١٩٢.١٦٨.١.١ <IP>"
