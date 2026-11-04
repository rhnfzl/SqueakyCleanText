"""Optional Model Context Protocol server.

Server settings come from the environment only:

``SCT_MCP_ROOT``
    Folder the ``path`` input may read from. Unset means ``path`` is off.
``SCT_MCP_MAX_BYTES``
    Largest file ``path`` may read (default 1 MiB).
``SCT_MCP_PRESET``
    Config preset for the built-in cleaners (default ``sct``, the library
    default; ``training`` keeps text as written apart from PII).
``SCT_MCP_TO_ASCII``
    ``1`` transliterates output to ASCII (0.7 default, ``日本語`` -> ``Ri Ben Yu``);
    unset keeps every script.
"""

import os
import stat
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, cast

from sct.adapters import _DuplicateKeyError, process_json
from sct.presets import preset_config
from sct.sct import TextCleaner
from sct.signals import Signals, public_view

if TYPE_CHECKING:
    from fastmcp.exceptions import ToolError
else:
    try:
        from fastmcp.exceptions import ToolError
    except ImportError:  # the helpers below stay usable (and testable) without the mcp extra
        class ToolError(Exception):
            """Stand-in for ``fastmcp.exceptions.ToolError``."""

DEFAULT_MAX_BYTES = 1_048_576
# path needs a dir-fd walk with O_NOFOLLOW; without it (Windows) path stays off.
PATH_SUPPORTED = os.open in os.supports_dir_fd and hasattr(os, "O_NOFOLLOW")


class Finding(TypedDict):
    """One detected entity, without its source text."""

    entity_type: str
    score: float
    start: int
    end: int
    action: str
    reason: str


class JsonFinding(Finding):
    path: list[str | int]


class TextOutput(TypedDict):
    warnings: list[str]
    summary: str
    text: str
    signals: Signals
    findings: NotRequired[list[Finding]]


class JsonOutput(TypedDict):
    warnings: list[str]
    summary: str
    value: Any
    signals: Signals
    findings: NotRequired[list[JsonFinding]]


def _public_finding(finding) -> Finding:
    """Serialize a finding without returning the source entity value."""
    return {
        "entity_type": finding.entity.entity_type,
        "score": finding.entity.score,
        "start": finding.entity.start,
        "end": finding.entity.end,
        "action": finding.action,
        "reason": finding.reason,
    }


def _full_warnings(signals: Signals, findings: list) -> list[str]:
    """Warn when entities were counted but the cleaner did not collect their findings."""
    if signals["entity_counts"] and not findings:
        return sorted({*signals["warnings"], "findings_not_collected"})
    return signals["warnings"]


def summarize(signals: Signals) -> str:
    """One value-free sentence, e.g. ``2 PII items (EMAIL, PERSON), 4% of text, language HINDI 0.93``."""
    count = signals["pii_count"]
    if count:
        pii = (
            f"{count} PII item{'s' if count != 1 else ''} "
            f"({', '.join(signals['pii_types'])}), {signals['pii_share']:.0%} of text"
        )
    else:
        pii = "No PII found"
    language = signals["language"]
    if language is None:
        return f"{pii}, language unknown"
    confidence = signals["language_confidence"]
    return f"{pii}, language {language}" + (f" {confidence:.2f}" if confidence is not None else "")


def sanitize_text(
    cleaner: TextCleaner,
    text: str,
    detail: Literal["concise", "full"] = "concise",
) -> TextOutput:
    """Return a JSON-safe, value-free text cleaning result."""
    result = cleaner.process(text)
    signals = public_view(result)
    output: TextOutput = {
        "warnings": signals["warnings"],
        "summary": summarize(signals),
        "text": result.lm_text,
        "signals": signals,
    }
    if detail == "full":
        output["findings"] = [
            _public_finding(finding)
            for finding in result.metadata.get("findings", ())
        ]
        output["warnings"] = _full_warnings(signals, output["findings"])
    return output


def sanitize_json(
    cleaner: TextCleaner,
    value: Any,
    detail: Literal["concise", "full"] = "concise",
) -> JsonOutput:
    """Clean every string in a JSON value; key findings use the ``@key`` index path."""
    try:
        result = process_json(cleaner, value)
    except _DuplicateKeyError as exc:
        raise ToolError(f"{exc}. Pass a JSON value whose keys stay unique after cleaning.") from None
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    signals = result.signals
    output: JsonOutput = {
        "warnings": signals["warnings"],
        "summary": summarize(signals),
        "value": result.value,
        "signals": signals,
    }
    if detail == "full":
        output["findings"] = [
            {"path": list(item.path), **_public_finding(item.finding)}
            for item in result.findings
        ]
        output["warnings"] = _full_warnings(signals, output["findings"])
    return output


def read_under_root(root: str, path: str, max_bytes: int = DEFAULT_MAX_BYTES) -> str:
    """Read a UTF-8 regular file below *root*, opening it folder by folder.

    Each component is opened relative to its parent's descriptor with
    ``O_NOFOLLOW``, so a symlink anywhere or a folder swapped mid-walk cannot
    lead outside *root*. Errors never echo the path or the file content.
    """
    if not PATH_SUPPORTED:
        raise ToolError("path input needs dir_fd and O_NOFOLLOW, which this platform lacks. Pass text instead.")
    parts = path.split("/")
    if os.path.isabs(path) or "\x00" in path or any(not p or p.startswith(".") for p in parts):
        raise ToolError(
            "path must be relative to SCT_MCP_ROOT, with no empty, hidden, '.' or '..' parts."
        )
    return _read_capped(_open_below(root, parts), max_bytes)


def _open_below(root: str, parts: list[str]) -> int:
    """Descriptor of the file *parts* names below *root*, each folder opened from its parent's descriptor."""
    try:
        dir_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        raise ToolError("SCT_MCP_ROOT cannot be opened. Check that the folder exists and is readable.") from None
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd)
            os.close(dir_fd)
            dir_fd = next_fd
        return os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
    except OSError:
        raise ToolError(
            "path not found under SCT_MCP_ROOT, or it goes through a symlink or a non-folder."
        ) from None
    finally:
        os.close(dir_fd)


def _read_capped(file_fd: int, max_bytes: int) -> str:
    """Read and close *file_fd*: a regular UTF-8 file of at most *max_bytes* bytes."""
    try:
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise ToolError("path must name a regular file, not a folder, link, pipe or device.")
        chunks, remaining = [], max_bytes + 1
        while remaining > 0:
            chunk = os.read(file_fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(file_fd)
    data = b"".join(chunks)
    if len(data) > max_bytes:
        raise ToolError(f"file is larger than {max_bytes} bytes. Raise SCT_MCP_MAX_BYTES or pass a smaller file.")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ToolError("file is not valid UTF-8. Convert it to UTF-8 or pass text instead.") from None


_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,  # a cold cache downloads NER models from the Hugging Face Hub
}


def _server_settings(root: str | None, max_bytes: int | None, preset: str | None) -> tuple[str | None, int, str]:
    """Arguments, else the environment, else the defaults; a bad value fails at start-up, not per call."""
    root = root or os.environ.get("SCT_MCP_ROOT") or None
    if root is not None and not os.path.isdir(root):
        raise ValueError("SCT_MCP_ROOT must be an existing directory")
    if max_bytes is None:
        max_bytes = int(os.environ.get("SCT_MCP_MAX_BYTES") or DEFAULT_MAX_BYTES)
    if max_bytes < 0:
        raise ValueError("SCT_MCP_MAX_BYTES must be zero or more")
    preset = preset or os.environ.get("SCT_MCP_PRESET") or "sct"
    preset_config(preset)
    return root, max_bytes, preset


def _cleaner_getter(cleaner: TextCleaner | None, preset: str) -> Callable[[bool], TextCleaner]:
    """``get_cleaner(pii)``: the given cleaner, else one built on first use for each mode."""
    ascii_override = {"check_to_ascii_unicode": True} if os.environ.get("SCT_MCP_TO_ASCII") == "1" else {}
    cleaners: dict[bool, TextCleaner] = {}
    cleaners_lock = threading.Lock()  # sync tools run on worker threads

    def get_cleaner(pii: bool) -> TextCleaner:
        if cleaner is not None:
            if pii and cleaner.cfg.ner_mode != "pii":
                raise ToolError(
                    "This server's cleaner is not configured for PII mode. "
                    "Call with pii=false, or start the server with a PII-mode cleaner."
                )
            return cleaner
        if pii not in cleaners:
            with cleaners_lock:
                if pii not in cleaners:
                    cleaners[pii] = TextCleaner(
                        cfg=preset_config(preset, ner_mode="pii" if pii else "standard", **ascii_override),
                        include_entities=True,
                    )
        return cleaners[pii]
    return get_cleaner


def create_mcp_server(
    cleaner: TextCleaner | None = None,
    *,
    root: str | None = None,
    max_bytes: int | None = None,
    preset: str | None = None,
) -> Any:
    """Create a FastMCP server without importing FastMCP at package import.

    *root*, *max_bytes* and *preset* default to ``SCT_MCP_ROOT`` /
    ``SCT_MCP_MAX_BYTES`` / ``SCT_MCP_PRESET``.
    """
    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "FastMCP is required for the MCP server. "
            "Install with: pip install squeakycleantext[mcp]"
        ) from exc

    root, max_bytes, preset = _server_settings(root, max_bytes, preset)

    server = FastMCP(
        "SqueakyCleanText",
        instructions=(
            "Remove or replace sensitive text before passing content to "
            "models, logs, or retrieval systems."
        ),
    )
    get_cleaner = _cleaner_getter(cleaner, preset)

    @server.tool(title="Clean text", annotations={**_ANNOTATIONS, "title": "Clean text"})
    def clean_text(
        text: str | None = None,
        path: str | None = None,
        pii: bool = False,
        detail: Literal["concise", "full"] = "concise",
    ) -> TextOutput:
        """Clean one text (or one UTF-8 file under SCT_MCP_ROOT) and return value-free signals.

        Pass exactly one of text or path. detail="full" adds per-entity
        findings (type, score, offsets) without the entity text.
        """
        if (text is None) == (path is None):
            raise ToolError("Pass exactly one of text or path.")
        if path is not None:
            if root is None:
                raise ToolError("path input is off. Set SCT_MCP_ROOT to a folder to allow it, or pass text.")
            text = read_under_root(root, path, max_bytes)
        return sanitize_text(get_cleaner(pii), cast(str, text), detail)

    @server.tool(title="Clean JSON", annotations={**_ANNOTATIONS, "title": "Clean JSON"})
    def clean_json(
        value: Any,
        pii: bool = False,
        detail: Literal["concise", "full"] = "concise",
    ) -> JsonOutput:
        """Clean every string (keys included) in a JSON value and return value-free signals."""
        return sanitize_json(get_cleaner(pii), value, detail)

    return server


def main() -> None:
    """Run the MCP server over standard input and output."""
    create_mcp_server().run()


if __name__ == "__main__":
    main()
