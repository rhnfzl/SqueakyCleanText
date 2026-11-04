import json

import pytest

from sct import TextCleanerConfig
from sct.browser import build_browser_manifest, write_browser_manifest
from sct.config import DEFAULT_NER_MODELS
import re
from sct import TextCleaner
from sct.utils import constants


def test_browser_manifest_pins_onnx_model_and_records_limitations():
    manifest = build_browser_manifest(TextCleanerConfig())

    assert manifest["format_version"] == 1
    assert manifest["models"]["ENGLISH"]["revision"] == (
        "2c10955af9538d6ff8a68316c5dbfcd0b968761e"
    )
    assert "language detection" in " ".join(manifest["limitations"]).lower()


def test_browser_manifest_rejects_unpinned_custom_models():
    models = dict(DEFAULT_NER_MODELS)
    models["ENGLISH"] = "example/custom-onnx-model"
    config = TextCleanerConfig(
        ner_models=models,
    )

    with pytest.raises(ValueError, match="ENGLISH"):
        build_browser_manifest(config)


def test_write_browser_manifest_creates_deterministic_json(tmp_path):
    output = tmp_path / "manifest.json"

    write_browser_manifest(TextCleanerConfig(), output)

    assert json.loads(output.read_text())["format_version"] == 1
    assert output.read_text().endswith("\n")


def test_browser_date_rule_follows_date_options():
    cfg = TextCleanerConfig(date_month_name_slash_year=False, date_builtin_month_languages=(),
                            check_ner_process=False)
    manifest = build_browser_manifest(cfg)
    assert not re.search(manifest["rules"]["date"]["pattern"], "Born October/55 here", re.I)
    assert TextCleaner(cfg=cfg).process("Born October/55 here").lm_text == "Born October/55 here"
    assert build_browser_manifest(TextCleanerConfig())["rules"]["date"]["pattern"] == constants.DATE_REGEX.pattern


def test_browser_phone_and_ip_rules_fold_digits():
    rules = build_browser_manifest(TextCleanerConfig())["rules"]
    assert rules["phone"]["fold_decimal_digits"] and rules["ip"]["fold_decimal_digits"]


def test_browser_date_rule_folds_digits():
    assert build_browser_manifest(TextCleanerConfig())["rules"]["date"]["fold_decimal_digits"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ner_backend": "torch"}, "onnx"),
        ({"replacement_mode": "synthetic"}, "placeholder"),
        ({"replace_ips_public_only": True}, "replace_ips_public_only"),
        ({"replace_with_email": "<EMAIL1>"}, "without digits"),
        ({"replace_with_url": "$URL"}, "currency"),
        ({"replace_with_html": "a@b.com"}, "later email rule"),
    ],
)
def test_browser_manifest_rejects_unportable_config(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_browser_manifest(TextCleanerConfig(**kwargs))


@pytest.mark.parametrize("mode", ["identifiers", "all"])
def test_browser_manifest_rules_follow_config(mode):
    config = TextCleanerConfig(replace_numbers_mode=mode)
    manifest = build_browser_manifest(config)

    assert manifest["rule_order"] == [
        "html", "url", "email", "ip", "date", "phone", "year", "number", "currency",
    ]
    for name, token, enabled in [
        ("html", config.replace_with_html, config.check_replace_html),
        ("url", config.replace_with_url, config.check_replace_urls),
        ("email", config.replace_with_email, config.check_replace_emails),
        ("ip", config.replace_with_ips, config.check_replace_ips),
        ("date", config.replace_with_dates, config.check_replace_dates),
        ("phone", config.replace_with_phone_numbers, config.check_replace_phone_numbers),
        ("year", config.replace_with_years, config.check_replace_years),
        ("number", config.replace_with_numbers, config.check_replace_numbers),
        ("currency", config.replace_with_currency_symbols, config.check_replace_currency_symbols),
    ]:
        assert manifest["rules"][name]["replacement"] == token
        assert manifest["rules"][name]["enabled"] == enabled

    folded = {name for name, rule in manifest["rules"].items() if rule.get("fold_decimal_digits")}
    assert folded == {"ip", "date", "phone"} | ({"number"} if mode == "identifiers" else set())
