"""Browser deployment manifest for ONNX Runtime Web integrations."""

import json
import re
from pathlib import Path
from typing import Any

from sct.config import TextCleanerConfig
from sct.utils import constants


def _rule(pattern, replacement, enabled: bool) -> dict[str, Any]:
    flags = "u"
    if pattern.flags & re.IGNORECASE:
        flags += "i"
    return {
        "pattern": pattern.pattern,
        "flags": flags,
        "replacement": replacement,
        "enabled": enabled,
    }


def build_browser_manifest(config: TextCleanerConfig) -> dict[str, Any]:
    """Describe portable rules and pinned ONNX assets for a browser client."""
    if config.ner_backend != "onnx":
        raise ValueError("Browser export currently supports ner_backend='onnx' only")
    if config.replacement_mode != "placeholder":
        raise ValueError(
            "Browser export currently supports placeholder replacement only"
        )

    model_revisions = config.ner_model_revisions or {}
    unpinned_models = sorted(
        key
        for key in (config.ner_models or {})
        if not model_revisions.get(key)
    )
    if unpinned_models:
        raise ValueError(
            "Browser export requires pinned revisions for models: "
            + ", ".join(unpinned_models)
        )

    models = {
        key: {
            "id": model_id,
            "revision": model_revisions[key],
        }
        for key, model_id in (config.ner_models or {}).items()
    }
    return {
        "format_version": 1,
        "runtime": "onnxruntime-web",
        "models": models,
        "rule_order": [
            "html", "url", "email", "date", "year", "phone", "number",
            "currency",
        ],
        "rules": {
            "html": _rule(
                constants.HTML_REGEX,
                config.replace_with_html,
                config.check_replace_html,
            ),
            "url": _rule(
                constants.URL_REGEX,
                config.replace_with_url,
                config.check_replace_urls,
            ),
            "email": _rule(
                constants.EMAIL_REGEX,
                config.replace_with_email,
                config.check_replace_emails,
            ),
            "date": _rule(
                constants.DATE_REGEX,
                config.replace_with_dates,
                config.check_replace_dates,
            ),
            "year": _rule(
                constants.YEAR_REGEX,
                config.replace_with_years,
                config.check_replace_years,
            ),
            "phone": _rule(
                constants.PHONE_REGEX,
                config.replace_with_phone_numbers,
                config.check_replace_phone_numbers,
            ),
            "number": _rule(
                constants.NUMBERS_REGEX,
                config.replace_with_numbers,
                config.check_replace_numbers,
            ),
            "currency": _rule(
                constants.CURRENCY_REGEX,
                config.replace_with_currency_symbols,
                config.check_replace_currency_symbols,
            ),
        },
        "limitations": [
            "Python language detection is not included in the browser bundle.",
            "Regular expressions require JavaScript compatibility tests.",
            "Model files remain external and must keep their recorded revisions.",
            "Reversible and synthetic replacement are server-side features.",
        ],
    }


def write_browser_manifest(
    config: TextCleanerConfig,
    output: str | Path,
) -> None:
    """Write a deterministic JSON deployment manifest."""
    Path(output).write_text(
        json.dumps(
            build_browser_manifest(config),
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
