"""Canonical findings validate stages without storing matched values."""

from dataclasses import FrozenInstanceError, asdict
import json

import pytest

from sct.findings import Detection, DetectionContext, TextRange


def test_findings_are_frozen_value_free_and_unknown_score_stays_unknown():
    finding = Detection("PERSON", 0, 3, "native", None, "normalized")
    finding.validate("👋 A", coordinate_space="normalized")
    assert finding.score is None
    assert "matched" not in json.dumps(asdict(finding))
    with pytest.raises(FrozenInstanceError):
        finding.start = 1
    with pytest.raises(ValueError):
        finding.validate("ab", coordinate_space="normalized")
    with pytest.raises(ValueError):
        finding.validate("abc", coordinate_space="original")


@pytest.mark.parametrize("start,end", [(0, 0), (-1, 2), (2, 1), (False, 2), (0, 1.5), (0, float("nan"))])
def test_invalid_ranges_fail(start, end):
    with pytest.raises(ValueError):
        TextRange(start, end)
    with pytest.raises(ValueError):
        Detection("PERSON", start, end, "native", None, "normalized")


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True, "0.5"])
def test_invalid_scores_fail(score):
    with pytest.raises(ValueError):
        Detection("PERSON", 0, 1, "native", score, "normalized")


def test_source_mapping_consistency():
    for quality, ranges in [("exact", ()), ("covering", ()), ("unavailable", (TextRange(0, 1),)), ("unknown", ())]:
        with pytest.raises(ValueError):
            Detection("PERSON", 0, 1, "native", 1.0, "normalized", ranges, quality)


def test_context_is_immutable_and_does_not_serialize_caller_data():
    context = DetectionContext("private-id", ("private-field", 1), ("PERSON",), "schema")
    assert "private" not in repr(context)
    assert context.to_dict() == {}
    with pytest.raises(FrozenInstanceError):
        context.path = ()
    with pytest.raises(ValueError):
        DetectionContext(path=([],))


def test_unbounded_integer_score_fails_validation():
    with pytest.raises(ValueError):
        Detection("PERSON", 0, 1, "native", 10**1000, "normalized")
