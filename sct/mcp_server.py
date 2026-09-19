"""Optional Model Context Protocol server."""

from typing import Any, Optional

from sct.adapters import process_json
from sct.config import TextCleanerConfig
from sct.sct import TextCleaner


def _public_finding(finding) -> dict[str, Any]:
    """Serialize a finding without returning the source entity value."""
    return {
        "entity": {
            "entity_type": finding.entity.entity_type,
            "score": finding.entity.score,
            "start": finding.entity.start,
            "end": finding.entity.end,
        },
        "action": finding.action,
        "reason": finding.reason,
    }


def sanitize_text(cleaner: TextCleaner, text: str) -> dict[str, Any]:
    """Return a JSON-safe text cleaning result."""
    result = cleaner.process(text)
    metadata = result.metadata or {}
    return {
        "text": result.lm_text,
        "stat_text": result.stat_text,
        "language": result.language,
        "findings": [
            _public_finding(finding)
            for finding in metadata.get("findings", ())
        ],
        "policy": metadata.get("policy"),
    }


def create_mcp_server(
    cleaner: Optional[TextCleaner] = None,
):
    """Create a FastMCP server without importing FastMCP at package import."""
    try:
        from fastmcp import FastMCP
    except ImportError as exc:
        raise ImportError(
            "FastMCP is required for the MCP server. "
            "Install with: pip install squeakycleantext[mcp]"
        ) from exc

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
                raise ValueError(
                    "The injected cleaner is not configured for PII mode"
                )
            return cleaner
        if pii not in cleaners:
            cleaners[pii] = TextCleaner(
                cfg=TextCleanerConfig(ner_mode="pii" if pii else "standard"),
                include_entities=True,
            )
        return cleaners[pii]

    @server.tool
    def clean_text(text: str, pii: bool = False) -> dict[str, Any]:
        """Clean one text value and return findings without logging source text."""
        return sanitize_text(get_cleaner(pii), text)

    @server.tool
    def clean_json(value: Any, pii: bool = False) -> dict[str, Any]:
        """Clean every string in a JSON-compatible value."""
        result = process_json(get_cleaner(pii), value)
        return {
            "value": result.value,
            "findings": [
                {
                    "path": list(item.path),
                    "finding": _public_finding(item.finding),
                }
                for item in result.findings
            ],
        }

    return server


def main() -> None:
    """Run the MCP server over standard input and output."""
    create_mcp_server().run()


if __name__ == "__main__":
    main()
