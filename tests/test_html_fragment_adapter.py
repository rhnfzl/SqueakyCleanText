"""Exercise HTML fragment preparation through the public unified cleaner."""

from dataclasses import replace

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.policy_v2 import HandlingRule, UnifiedPrivacyPolicy


def config(**overrides):
    settings = {
        "pii_processing": "unified",
        "check_detect_language": False,
        "check_ner_process": False,
        "check_statistical_model_processing": False,
        "check_replace_numbers": False,
        "source_mapping": "require_source",
    }
    return TextCleanerConfig(**(settings | overrides))


def test_fragment_masks_split_email_and_preserves_cell_boundaries():
    html = (
        '<p>Reply 🧪 to <em\n title="x > y">lark<span>7</span></em>@example.test today.</p>'
        '<table><tr><th>Key</th><th>Domain</th></tr><tr><td>deskKey</td><td>@stock.test</td></tr></table>'
    )
    policy = UnifiedPrivacyPolicy("example", "1", {"EMAIL": HandlingRule(action="redact")})
    cleaner = TextCleaner(config(html_fragment_adapter=True), unified_policy=policy)

    result = cleaner.process(html)

    assert result.lm_text == "Reply 🧪 to <REDACTED> today.\nKey\nDomain\ndeskKey\n@stock.test"
    finding, = result.metadata["findings_v2"]
    assert finding["entity_type"] == "EMAIL"
    assert finding["action"] == "redact"
    assert finding["mapping_quality"] == "exact"
    assert finding["source_ranges"] == [
        {"start": start, "end": start + len(part)}
        for part, start in (("lark", html.index("lark")), ("7", html.index(">7") + 1),
                            ("@example.test", html.index("@example.test")))
    ]


def test_fragment_reports_covering_map_when_unicode_repair_changes_alignment():
    html = (
        "<p>Cafe\u0301 &lt;i&gt; <i>Ｆinch</i>&#64;example.test done.</p>"
        "<table><tr><td>partKey</td><td>@stock.test</td></tr></table>"
    )

    result = TextCleaner(config(html_fragment_adapter=True)).process(html)

    assert result.lm_text == "Café <i> <EMAIL> done.\npartKey\n@stock.test"
    finding, = result.metadata["findings_v2"]
    assert finding["mapping_quality"] == "covering"
    assert finding["source_ranges"] == [
        {"start": 3, "end": 19}, {"start": 22, "end": 27}, {"start": 31, "end": 100},
    ]


@pytest.mark.parametrize("html", [
    "<p>Good <b>part</b></p><a href='mailto:scratch@example.test'>other</a>",
    "<p>Good</p><p><!-- comment --></p>",
    "<p>Good</p><br/>",
    "<p>Good <b>part</p>",
    "<p>Good</p><b",
    "<p>Good</p></>",
    "<p>Good</p><b / title=x>other</b>",
    "<table><td>wrong parent</td></table>",
    "<p><div>wrong parent</div></p>",
    "<p>Good</p><!unsupported>",
    "Plain input  with a@example.test",
])
@pytest.mark.parametrize("replace_html", [True, False])
def test_unsupported_fragment_uses_original_fixed_settings(html, replace_html):
    fixed = config(check_replace_html=replace_html, replace_with_html=" ")
    selected = replace(fixed, html_fragment_adapter=True)

    assert TextCleaner(selected).process(html) == TextCleaner(fixed).process(html)


def test_one_space_html_replacement_remains_available_without_adapter():
    cleaner = TextCleaner(config(replace_with_html=" "))

    assert cleaner.process("<p>Reply to <b>someone</b> today.</p>").lm_text == "Reply to someone today."


def test_fragment_adapter_requires_explicit_unified_selection():
    with pytest.raises(ValueError, match="requires unified"):
        TextCleanerConfig(html_fragment_adapter=True)
    with pytest.raises(ValueError, match="must be a boolean"):
        config(html_fragment_adapter="yes")
