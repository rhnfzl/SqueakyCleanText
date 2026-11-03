"""Browser deployment manifest for ONNX Runtime Web integrations."""

import json
import re
from pathlib import Path
from typing import Any

from sct.config import TextCleanerConfig
from sct.utils import constants, resources
from sct.utils.contact import config_numbers_regex


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


_JS_WORD = r"\p{L}\p{N}_"  # Python's Unicode \w; JavaScript's \w is ASCII even with the "u" flag


def _unicode_word(pattern: str) -> str:
    """Rewrite each \\w of a Python pattern as an explicit Unicode class, so "é" stays a word character in JS."""
    out, in_class, i = [], False, 0
    while i < len(pattern):
        part = pattern[i:i + 2 if pattern[i] == "\\" else i + 1]
        if part == r"\w":
            part = _JS_WORD if in_class else f"[{_JS_WORD}]"
        elif part == "[" and not in_class:
            in_class = True
        elif part == "]" and in_class:
            in_class = False
        out.append(part)
        i += 2 if pattern[i] == "\\" else 1
    return "".join(out)


def build_browser_manifest(config: TextCleanerConfig) -> dict[str, Any]:
    """Describe portable rules and pinned ONNX assets for a browser client."""
    if config.ner_backend != "onnx":
        raise ValueError("Browser export currently supports ner_backend='onnx' only")
    if config.replacement_mode != "placeholder":
        raise ValueError(
            "Browser export currently supports placeholder replacement only"
        )

    if config.check_replace_ips and config.replace_ips_public_only:
        raise ValueError(
            "Browser export does not support replace_ips_public_only=True"
        )

    # The browser runs rules in order with nothing protecting emitted tokens,
    # so a digit in one ("<EMAIL1>") would be rewritten by the number rules.
    tokens = (
        config.replace_with_html, config.replace_with_url, config.replace_with_email,
        config.replace_with_ips, config.replace_with_dates,
        config.replace_with_phone_numbers, config.replace_with_years,
    )
    if any(ch.isdigit() for token in tokens for ch in token):
        raise ValueError("Browser export requires replacement tokens without digits")
    if config.check_replace_currency_symbols and any(
        constants.CURRENCY_REGEX.search(token) for token in tokens
    ):
        raise ValueError(
            "Browser export requires replacement tokens without currency symbols"
        )

    # Same reason for any token a later enabled rule would match
    # (an HTML token holding an email or URL, a URL token holding an email).
    html_regex = constants.HTML_REGEX if config.html_tag_strict else constants.HTML_REGEX_ANY
    url_regex = constants.URL_REGEX if config.url_trailing_punct_outside else constants.URL_REGEX_LEGACY
    month_names = resources.configured_month_names(config)
    date_regex = (constants.DATE_REGEX if month_names is None
                  else constants.build_date_regex(month_names, slash_year=config.date_month_name_slash_year))
    number_regex = config_numbers_regex(config)
    ordered = [
        ("html", html_regex, config.replace_with_html, config.check_replace_html),
        ("url", url_regex, config.replace_with_url, config.check_replace_urls),
        ("email", constants.EMAIL_REGEX, config.replace_with_email, config.check_replace_emails),
        ("ip", constants.IP_REGEX, config.replace_with_ips, config.check_replace_ips),
        ("date", date_regex, config.replace_with_dates, config.check_replace_dates),
        ("phone", constants.PHONE_REGEX, config.replace_with_phone_numbers,
         config.check_replace_phone_numbers),
        ("year", constants.YEAR_REGEX, config.replace_with_years, config.check_replace_years),
        ("number", number_regex, config.replace_with_numbers,
         config.check_replace_numbers),
        ("currency", constants.CURRENCY_REGEX, config.replace_with_currency_symbols,
         config.check_replace_currency_symbols),
    ]
    for i, (name, _, token, enabled) in enumerate(ordered):
        for later, pattern, _, later_enabled in ordered[i + 1:]:
            if enabled and later_enabled and pattern.search(token):
                raise ValueError(
                    f"Browser export: the {name} replacement token would be "
                    f"rewritten by the later {later} rule"
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
            "html", "url", "email", "ip", "date", "phone", "year", "number",
            "currency",
        ],
        "rules": {
            "html": _rule(
                html_regex,
                config.replace_with_html,
                config.check_replace_html,
            ),
            "url": _rule(
                url_regex,
                config.replace_with_url,
                config.check_replace_urls,
            ),
            "email": _rule(
                constants.EMAIL_REGEX,
                config.replace_with_email,
                config.check_replace_emails,
            ),
            "ip": {**_rule(
                constants.IP_REGEX,
                config.replace_with_ips,
                config.check_replace_ips,
            ), "fold_decimal_digits": True},
            "date": {**_rule(
                date_regex,
                config.replace_with_dates,
                config.check_replace_dates,
            ), "fold_decimal_digits": True},
            "year": _rule(
                constants.YEAR_REGEX,
                config.replace_with_years,
                config.check_replace_years,
            ),
            "phone": {**_rule(
                constants.PHONE_REGEX,
                config.replace_with_phone_numbers,
                config.check_replace_phone_numbers,
            ), "fold_decimal_digits": True},
            # identifiers mode: \d of the pattern is Unicode in Python, so the browser reads ASCII-folded digits
            "number": {**_rule(
                number_regex,
                config.replace_with_numbers,
                config.check_replace_numbers,
            ), **({"fold_decimal_digits": True,
                   "pattern": _unicode_word(number_regex.pattern)}
                  if config.replace_numbers_mode == "identifiers" else {})},
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
            "IBAN/card checksum typing and identifier tokens are server-side rules.",
            "phone_reject_embedded is a server-side rule: the browser phone rule keeps every phone match "
            "(0.8.1 behaviour).",
            "custom_deny_terms, custom_allow_terms and custom_patterns are server-side features.",
            "The browser year rule runs before the number rule on its own, so a year-like group of an ID "
            "run ('123-45 2020') becomes the year token and the rest of the run stays; the server masks the "
            "whole run.",
            "Rules with fold_decimal_digits match on a copy of the text where every Unicode "
            "decimal digit (Nd, such as ٠٦ or ०६) is its ASCII digit, one code point for one, "
            "and replace the same offsets in the original text (JavaScript \\d is ASCII only).",
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
