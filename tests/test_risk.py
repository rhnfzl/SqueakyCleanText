from sct.risk import assess_reidentification_risk


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
