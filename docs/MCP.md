# MCP server

```bash
pip install 'SqueakyCleanText[mcp]'
sct-mcp                       # stdio server
```

## Tools

Both tools are read-only, non-destructive, idempotent and closed-world
(`readOnlyHint`, `destructiveHint=false`, `idempotentHint`, `openWorldHint=false`),
and both declare an `outputSchema`.

| Tool | Arguments |
|---|---|
| `clean_text` | `text` or `path` (exactly one), `pii=false`, `detail="concise"` |
| `clean_json` | `value`, `pii=false`, `detail="concise"` |

## Output

Keys, in this order:

| key | meaning |
|---|---|
| `warnings` | sorted warning codes (`language_unknown`, `language_fallback`, `language_low_confidence`, `ner_fallback_models`, `ner_off`, `cuda_fallback`, `empty_input`) |
| `summary` | one sentence, e.g. `2 PII items (EMAIL, PERSON), 4% of text, language HINDI 0.93` |
| `text` / `value` | the cleaned text, or the cleaned JSON value (keys cleaned too) |
| `signals` | the value-free public view: language, confidence, chars, PII counts and share, per-type counts, regex counts, models |
| `findings` | only with `detail="full"`: `entity_type, score, start, end, action, reason` per entity (plus `path` for JSON). Never the entity text |

There is no `stat_text`. For JSON, a finding inside a key uses the path
`[..., "@key", <index>]`, so the raw key never appears.

## Reading files (`path`)

Off by default. Set `SCT_MCP_ROOT` to a folder to let `clean_text` read files
under it:

```bash
SCT_MCP_ROOT=~/notes SCT_MCP_MAX_BYTES=2097152 sct-mcp
```

- `path` is relative to the root. Absolute paths, empty parts, `.`/`..` and any
  part starting with `.` (hidden files and folders) are refused.
- The file is opened folder by folder from the root's descriptor with
  `O_NOFOLLOW`: a symlink anywhere on the way, or a folder swapped during the
  walk, cannot lead outside the root.
- Only regular files (no folders, pipes or devices), at most `SCT_MCP_MAX_BYTES`
  bytes (default 1 MiB), strict UTF-8.
- Errors say what to fix and never echo the path's target or the file content.
- On platforms without `dir_fd` and `O_NOFOLLOW` (Windows) `path` stays off.

## Errors

Bad input raises an MCP `ToolError` with a remedy: both or neither of
`text`/`path`, `path` while `SCT_MCP_ROOT` is unset, `pii=true` against a
cleaner that is not in PII mode, JSON keys that collide after cleaning, and every
file error above.

## In Python

```python
from sct.mcp_server import create_mcp_server
server = create_mcp_server(cleaner=None, root="/data/in", max_bytes=1 << 20)
```

`root` and `max_bytes` default to the environment variables.
