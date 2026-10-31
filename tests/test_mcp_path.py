"""Security tests for the MCP ``path`` input (dir-fd walk under SCT_MCP_ROOT)."""

import asyncio
import os

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.mcp_server import PATH_SUPPORTED, ToolError, create_mcp_server, read_under_root

pytestmark = pytest.mark.skipif(not PATH_SUPPORTED, reason="needs dir_fd and O_NOFOLLOW")


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "root"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "a.txt").write_text("Mail maria@example.com", encoding="utf-8")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "b.txt").write_text("x", encoding="utf-8")
    (root / "docs" / ".secret.txt").write_text("x", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
    (root / "link.txt").symlink_to(outside / "secret.txt")
    (root / "linkdir").symlink_to(outside)
    (root / "docs" / "latin1.txt").write_bytes(b"caf\xe9")
    return root


def test_normal_file_is_read(tree):
    assert read_under_root(str(tree), "docs/a.txt") == "Mail maria@example.com"


@pytest.mark.parametrize("path", [
    "link.txt",                 # symlink file
    "linkdir/secret.txt",       # symlink dir in the middle
    ".hidden/b.txt",            # hidden dir
    "docs/.secret.txt",         # hidden file
    "../outside/secret.txt",    # parent escape
    "docs/../a.txt",
    "./docs/a.txt",
    "",                         # empty
    "docs//a.txt",              # empty component
    "docs/",                    # trailing slash
    "docs",                     # directory
    "missing.txt",
    "docs/a.txt/x",             # file used as a folder
])
def test_rejected_paths(tree, path):
    with pytest.raises(ToolError) as err:
        read_under_root(str(tree), path)
    assert "TOP SECRET" not in str(err.value)
    assert "outside" not in str(err.value)


def test_absolute_path_rejected_even_inside_root(tree):
    with pytest.raises(ToolError):
        read_under_root(str(tree), str(tree / "docs" / "a.txt"))


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs mkfifo")
def test_fifo_rejected_without_blocking(tree):
    os.mkfifo(tree / "pipe")
    with pytest.raises(ToolError, match="regular file"):
        read_under_root(str(tree), "pipe")


def test_size_cap(tree):
    (tree / "big.txt").write_bytes(b"a" * 11)
    assert read_under_root(str(tree), "big.txt", max_bytes=11) == "a" * 11
    with pytest.raises(ToolError, match="larger than 10 bytes"):
        read_under_root(str(tree), "big.txt", max_bytes=10)


def test_non_utf8_rejected_without_echoing_content(tree):
    with pytest.raises(ToolError, match="UTF-8") as err:
        read_under_root(str(tree), "docs/latin1.txt")
    assert "caf" not in str(err.value)
    assert "xe9" not in str(err.value)


def _call(server, **arguments):
    from fastmcp import Client

    async def run():
        async with Client(server) as client:
            return await client.call_tool("clean_text", arguments)

    return asyncio.run(run())


def _cleaner():
    return TextCleaner(cfg=TextCleanerConfig(
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
    ))


def test_path_off_unless_root_set(tree, monkeypatch):
    pytest.importorskip("fastmcp")
    from fastmcp.exceptions import ToolError as FastToolError

    monkeypatch.delenv("SCT_MCP_ROOT", raising=False)
    with pytest.raises(FastToolError, match="SCT_MCP_ROOT"):
        _call(create_mcp_server(_cleaner()), path="docs/a.txt")


def test_path_tool_reads_under_env_root(tree, monkeypatch):
    pytest.importorskip("fastmcp")
    from fastmcp.exceptions import ToolError as FastToolError

    monkeypatch.setenv("SCT_MCP_ROOT", str(tree))
    monkeypatch.setenv("SCT_MCP_MAX_BYTES", "5")
    server = create_mcp_server(_cleaner())
    with pytest.raises(FastToolError, match="larger than 5 bytes"):
        _call(server, path="docs/a.txt")

    server = create_mcp_server(_cleaner(), max_bytes=100)
    assert _call(server, path="docs/a.txt").structured_content["text"] == "Mail <EMAIL>"
    with pytest.raises(FastToolError, match="exactly one"):
        _call(server, path="docs/a.txt", text="x")
    with pytest.raises(FastToolError, match="exactly one"):
        _call(server)
    with pytest.raises(FastToolError) as err:
        _call(server, path="link.txt")
    assert "TOP SECRET" not in str(err.value)
