"""Optional Model Context Protocol server.

Server settings come from the environment only:

``SCT_MCP_ROOT``
    Folder the ``path`` input may read from. Unset means ``path`` is off.
``SCT_MCP_MAX_BYTES``
    Largest file ``path`` may read (default 1 MiB).
"""

import os
import stat
from typing import Any, Literal, NotRequired, Optional, TypedDict

from sct.adapters import process_json
from sct.config import TextCleanerConfig
from sct.sct import TextCleaner

try:
    from fastmcp.exceptions import ToolError
except ImportError:  # the helpers below stay usable (and testable) without the mcp extra
    class ToolError(Exception):
        """Stand-in for ``fastmcp.exceptions.ToolError``."""

try:
    from sct.signals import merge_views, public_view
except ImportError:  # shortcut: pre-K1 fallback built from metadata, delete once sct/signals.py is merged
    merge_views = public_view = None

DEFAULT_MAX_BYTES = 1_048_576
PII_REGEX_TYPES = frozenset({"EMAIL", "URL", "IP_ADDRESS", "PHONE_NUMBER"})
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
    signals: dict[str, Any]
    findings: NotRequired[list[Finding]]


class JsonOutput(TypedDict):
    warnings: list[str]
    summary: str
    value: Any
    signals: dict[str, Any]
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


def _fallback_view(result, chars: int) -> dict[str, Any]:
    metadata = result.metadata or {}
    regex_counts = {k: dict(v) for k, v in sorted(metadata.get("regex_counts", {}).items())}
    entity_counts = {k: dict(v) for k, v in sorted(metadata.get("entity_counts", {}).items())}
    pii_types: dict[str, int] = {}
    pii_chars = 0
    pii_items = [(k, v) for k, v in regex_counts.items() if k in PII_REGEX_TYPES]
    for name, entry in pii_items + list(entity_counts.items()):
        pii_types[name] = pii_types.get(name, 0) + entry["count"]
        pii_chars += entry["chars"]
    provenance = metadata.get("provenance") or {}
    warnings = set(metadata.get("warnings", ()))
    if not provenance.get("backend"):
        warnings.add("ner_off")
    if chars == 0:
        warnings.add("empty_input")
    return {
        "language": result.language,
        "language_confidence": metadata.get("language_confidence"),
        "language_source": metadata.get("language_source", "none"),
        "chars": chars,
        "pii_count": sum(pii_types.values()),
        "pii_chars": pii_chars,
        "pii_share": min(1.0, pii_chars / chars) if chars else 0.0,
        "pii_types": dict(sorted(pii_types.items())),
        "regex_counts": regex_counts,
        "entity_counts": entity_counts,
        "sensitive": bool(regex_counts or entity_counts),
        "warnings": sorted(warnings),
        "models": [
            {k: model.get(k) for k in ("key", "id", "revision", "provider")}
            for model in provenance.get("models", ())
        ],
    }


def _fallback_merge(views: list[dict[str, Any]]) -> dict[str, Any]:
    if not views:
        return _fallback_view(type("Empty", (), {"metadata": {}, "language": None})(), 0)
    # Start from the longest leaf so language_source follows it.
    merged = dict(max(views, key=lambda v: v["chars"]))
    lang_chars: dict[Any, int] = {}
    for view in views:
        lang_chars[view["language"]] = lang_chars.get(view["language"], 0) + view["chars"]
    language = max(lang_chars, key=lang_chars.get)
    same = [v for v in views if v["language"] == language and v["language_confidence"] is not None]
    weight = sum(v["chars"] for v in same)
    merged["language"] = language
    merged["language_confidence"] = (
        sum(v["language_confidence"] * v["chars"] for v in same) / weight if weight else None
    )
    for key in ("chars", "pii_count", "pii_chars"):
        merged[key] = sum(v[key] for v in views)
    for key in ("regex_counts", "entity_counts"):
        totals: dict[str, dict[str, int]] = {}
        for view in views:
            for name, entry in view[key].items():
                slot = totals.setdefault(name, {"count": 0, "chars": 0})
                slot["count"] += entry["count"]
                slot["chars"] += entry["chars"]
        merged[key] = dict(sorted(totals.items()))
    types: dict[str, int] = {}
    for view in views:
        for name, count in view["pii_types"].items():
            types[name] = types.get(name, 0) + count
    merged["pii_types"] = dict(sorted(types.items()))
    merged["pii_share"] = min(1.0, merged["pii_chars"] / merged["chars"]) if merged["chars"] else 0.0
    merged["sensitive"] = any(v["sensitive"] for v in views)
    merged["warnings"] = sorted({w for v in views for w in v["warnings"]})
    merged["models"] = list({m["key"]: m for v in views for m in v["models"]}.values())
    return merged


def _signals(result, chars: int) -> dict[str, Any]:
    return public_view(result) if public_view else _fallback_view(result, chars)


def _merge(views: list[dict[str, Any]]) -> dict[str, Any]:
    return merge_views(views) if merge_views else _fallback_merge(views)


def summarize(signals: dict[str, Any]) -> str:
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
    signals = _signals(result, len(text))
    output: TextOutput = {
        "warnings": signals["warnings"],
        "summary": summarize(signals),
        "text": result.lm_text,
        "signals": signals,
    }
    if detail == "full":
        output["findings"] = [
            _public_finding(finding)
            for finding in (result.metadata or {}).get("findings", ())
        ]
    return output


def sanitize_json(
    cleaner: TextCleaner,
    value: Any,
    detail: Literal["concise", "full"] = "concise",
) -> JsonOutput:
    """Clean every string in a JSON value; key findings use the ``@key`` index path."""

    class Recorder:
        cfg = cleaner.cfg

        def process(self, text):
            result = cleaner.process(text)
            views.append(_signals(result, len(text)))
            return result

    views: list[dict[str, Any]] = []
    try:
        result = process_json(Recorder(), value)
    except ValueError as exc:
        raise ToolError(f"{exc}. Pass a JSON value whose keys stay unique after cleaning.") from None
    signals = _merge(views)
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
    dir_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd)
            os.close(dir_fd)
            dir_fd = next_fd
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
    except OSError:
        raise ToolError(
            "path not found under SCT_MCP_ROOT, or it goes through a symlink or a non-folder."
        ) from None
    finally:
        os.close(dir_fd)
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
    "openWorldHint": False,
}


def create_mcp_server(
    cleaner: Optional[TextCleaner] = None,
    *,
    root: Optional[str] = None,
    max_bytes: Optional[int] = None,
):
    """Create a FastMCP server without importing FastMCP at package import.

    *root* and *max_bytes* default to ``SCT_MCP_ROOT`` / ``SCT_MCP_MAX_BYTES``.
    """
    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "FastMCP is required for the MCP server. "
            "Install with: pip install squeakycleantext[mcp]"
        ) from exc

    root = root or os.environ.get("SCT_MCP_ROOT") or None
    if max_bytes is None:
        max_bytes = int(os.environ.get("SCT_MCP_MAX_BYTES") or DEFAULT_MAX_BYTES)
    if max_bytes < 0:
        raise ValueError("SCT_MCP_MAX_BYTES must be zero or more")

    server = FastMCP(
        "SqueakyCleanText",
        instructions=(
            "Remove or replace sensitive text before passing content to "
            "models, logs, or retrieval systems."
        ),
    )
    cleaners: dict[bool, TextCleaner] = {}

    def get_cleaner(pii: bool) -> TextCleaner:
        if cleaner is not None:
            if pii and cleaner.cfg.ner_mode != "pii":
                raise ToolError(
                    "This server's cleaner is not configured for PII mode. "
                    "Call with pii=false, or start the server with a PII-mode cleaner."
                )
            return cleaner
        if pii not in cleaners:
            cleaners[pii] = TextCleaner(
                cfg=TextCleanerConfig(ner_mode="pii" if pii else "standard"),
                include_entities=True,
            )
        return cleaners[pii]

    @server.tool(title="Clean text", annotations={**_ANNOTATIONS, "title": "Clean text"})
    def clean_text(
        text: Optional[str] = None,
        path: Optional[str] = None,
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
        return sanitize_text(get_cleaner(pii), text, detail)

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
