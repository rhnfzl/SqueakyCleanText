"""Source provenance remains honest across explicit and opaque edits."""

import json

import pytest

from sct.coordinates import SourceMap
from sct.findings import TextRange


def test_c1_acceptance():
    mapping = SourceMap.identity("👋 A&amp;B").replace(3, 8, "&")
    assert mapping.project(4, 5).ranges == (TextRange(8, 9),)
    assert mapping.project(4, 5).mapping_quality == "exact"
    assert mapping.project(3, 4).ranges == (TextRange(3, 8),)
    assert mapping.project(3, 4).mapping_quality == "covering"


def test_insert_delete_and_expand_compose():
    original = SourceMap.identity("abcd")
    assert original.project(0, 4).ranges == (TextRange(0, 4),)
    deleted = original.replace(1, 2, "")
    assert deleted.project(0, 3).ranges == (TextRange(0, 1), TextRange(2, 4))
    assert deleted.project(0, 3).mapping_quality == "exact"
    expanded = deleted.replace(1, 2, "XY")
    assert expanded.project(1, 2).ranges == (TextRange(2, 3),)
    assert expanded.project(1, 2).mapping_quality == "covering"
    inserted = expanded.replace(0, 0, "!")
    assert inserted.project(0, 1).mapping_quality == "unavailable"
    assert inserted.project(0, 2).ranges == ()
    assert inserted.project(1, 2).mapping_quality == "exact"
    assert original.project(1, 2).mapping_quality == "exact"


def test_opaque_transform_never_recovers_provenance():
    mapping = SourceMap.identity("private value").transform(lambda text: text[::-1])
    assert mapping.project(0, 3).mapping_quality == "unavailable"
    assert mapping.replace(0, 1, "X").project(0, 1).ranges == ()
    assert "private" not in repr(mapping)
    assert "eulav" not in json.dumps(mapping.to_dict())


@pytest.mark.parametrize("start,end", [(-1, 1), (0, 4), (2, 1), (True, 2), (0, float("nan")), (0, float("inf"))])
def test_invalid_stage_offsets_fail(start, end):
    mapping = SourceMap.identity("abc")
    with pytest.raises(ValueError):
        mapping.project(start, end)
    with pytest.raises(ValueError):
        mapping.replace(start, end, "x")


def test_empty_edits_and_noop_preserve_provenance():
    mapping = SourceMap.identity("").replace(0, 0, "abc")
    assert mapping.project(0, 3).mapping_quality == "unavailable"
    mapping = SourceMap.identity("abc").replace(0, 3, "abc")
    assert mapping.project(0, 3).mapping_quality == "exact"
    with pytest.raises(ValueError):
        mapping.project(1, 1)


def test_projection_validation_and_source_map_immutability():
    from sct.coordinates import Projection

    with pytest.raises(ValueError):
        Projection((TextRange(2, 3), TextRange(1, 2)), "exact")
    with pytest.raises(ValueError):
        Projection((), "exact")
    mapping = SourceMap.identity("private")
    with pytest.raises(AttributeError):
        mapping._text = "changed"
    with pytest.raises(AttributeError):
        del mapping._text


def test_invalid_transform_and_replacement_fail():
    mapping = SourceMap.identity("abc")
    with pytest.raises(ValueError):
        mapping.transform(lambda text: None)
    with pytest.raises(ValueError):
        mapping.replace(0, 1, None)


def test_private_isolation_accepts_local_cover_and_rejects_shared_stage_cover():
    local = SourceMap.identity('😀Alice\nExample tail').replace(6, 7, ' ')
    assert local.project(1, 14).mapping_quality == 'covering'
    assert local._is_isolated(1, 14)
    broad = SourceMap.identity('😀Alice tail').replace(0, 11, 'Alice tail')
    assert broad.project(0, 5).mapping_quality == 'covering'
    assert not broad._is_isolated(0, 5)
    assert broad._is_isolated(0, 10)


def test_private_isolation_rejects_opaque_and_nonmonotone_maps():
    from sct.coordinates import Projection
    from sct.findings import TextRange
    opaque = SourceMap.identity('ab').transform(str.lower)
    assert not opaque._is_isolated(0, 1)
    nonmonotone = SourceMap('ab', (Projection((TextRange(1, 2),), 'covering'),
                                  Projection((TextRange(0, 1),), 'covering')))
    assert not nonmonotone._is_isolated(0, 1)
    assert nonmonotone.project(0, 1).mapping_quality == 'covering'
