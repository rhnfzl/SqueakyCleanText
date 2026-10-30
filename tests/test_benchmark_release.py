from scripts.benchmark_release import find_leaks, predicted_spans
from sct.evaluation import EntitySpan, EvaluationExample


def test_predicted_spans_maps_placeholders_in_order():
    source = "Call 555 1234, then 555 1234 again; Ann knows."
    output = "Call <NUMBER_0>, then <NUMBER_0> again; <PERSON_0> knows. <EMAIL_0>"
    originals = {"<NUMBER_0>": "555 1234", "<PERSON_0>": "Ann", "<EMAIL_0>": "gone@x.io"}

    assert predicted_spans(source, output, originals) == [
        EntitySpan("NUMBER", 5, 13),
        EntitySpan("NUMBER", 20, 28),
        EntitySpan("PERSON", 36, 39),
        EntitySpan("EMAIL", -1, -1),
    ]


def test_find_leaks_reports_full_and_partial_pii_only():
    text = "Bram de Vries paid 2023 from 8.8.8.8"
    example = EvaluationExample(
        id="doc",
        text=text,
        language="nl",
        domain="test",
        expected=(
            EntitySpan("PERSON", 0, 13),
            EntitySpan("YEAR", 19, 23),
            EntitySpan("IP", 29, 36),
        ),
    )

    leaks, partial = find_leaks(example, "<PERSON> Vries paid 2023 from 8.8.8.8")

    assert leaks == [{"id": "doc", "label": "IP", "value": "8.8.8.8"}]
    assert partial == [{"id": "doc", "label": "PERSON", "value": "Bram de Vries"}]


def test_predicted_spans_skip_untouched_earlier_copy():
    assert predicted_spans("Ann met Ann", "Ann met <PERSON_0>", {"<PERSON_0>": "Ann"}) == [
        EntitySpan("PERSON", 8, 11),
    ]


def test_predicted_spans_map_ip_address_to_dataset_label():
    assert predicted_spans("at 8.8.8.8", "at <IP_ADDRESS_0>", {"<IP_ADDRESS_0>": "8.8.8.8"}) == [
        EntitySpan("IP", 3, 10),
    ]


def test_find_leaks_catches_split_values():
    text = "Call 9876543210 from 192.168.1.10"
    example = EvaluationExample(
        id="doc", text=text, language="en", domain="test",
        expected=(EntitySpan("PHONE_NUMBER", 5, 15), EntitySpan("IP", 21, 33)),
    )

    leaks, partial = find_leaks(example, "Call 98765<NUMBER> from <NUMBER_0>.1.10")

    assert leaks == []
    assert [item["label"] for item in partial] == ["PHONE_NUMBER", "IP"]


def test_predicted_spans_context_changed_by_cleanup():
    # Isolated-letter removal drops "x", so the output context is not in the source.
    assert predicted_spans("Ann x met Ann", "Ann met <PERSON_0>", {"<PERSON_0>": "Ann"}) == [
        EntitySpan("PERSON", 10, 13),
    ]
