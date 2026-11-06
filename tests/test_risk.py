from sct.risk import assess_reidentification_risk
import pytest


def test_reidentification_report_counts_small_equivalence_classes():
    records = (
        {"city": "Utrecht", "age_band": "30-39"},
        {"city": "Utrecht", "age_band": "30-39"},
        {"city": "Delft", "age_band": "50-59"},
    )

    report = assess_reidentification_risk(
        records,
        quasi_identifiers=("city", "age_band"),
        k_threshold=2,
    )

    assert report.record_count == 3
    assert report.minimum_k == 1
    assert report.at_risk_records == 1
    assert report.at_risk_fraction == 1 / 3
    assert not hasattr(report, "values")


def test_risk_bool_and_int_are_distinct():
    risk = assess_reidentification_risk([{"q": True}, {"q": 1}], quasi_identifiers=("q",), k_threshold=2)
    assert risk.minimum_k == 1


def test_a_string_quasi_identifier_names_one_field():
    risk = assess_reidentification_risk([{"age": 30}, {"age": 30}], quasi_identifiers="age", k_threshold=2)
    assert risk.quasi_identifiers == ("age",)
    assert risk.record_count == 2


# --- type tags reach inside tuple and frozenset quasi-identifiers ---
@pytest.mark.parametrize("a, b", [((True,), (1,)), ((1,), (1.0,)),
                                  (frozenset({True}), frozenset({1})), (((1,),), ((True,),))])
def test_nested_quasi_identifiers_are_type_tagged(a, b):
    report = assess_reidentification_risk([{"q": a}, {"q": b}], quasi_identifiers=("q",),
                                           k_threshold=2)
    assert (report.minimum_k, report.at_risk_records) == (1, 2)
