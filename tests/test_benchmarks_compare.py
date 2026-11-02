"""Tool-comparison harness: dev/test split, new loaders' span building, entry tables (no dataset data)."""

from benchmarks.compare import entries, split
from benchmarks.datasets import MODULES, get
from benchmarks.datasets.gretel_en import entity_spans
from benchmarks.labels import DATASET_LABELS, normalise_gold


def rec(rid, group=None):
    r = {"id": rid, "lang": "en", "text": "x", "spans": []}
    if group is not None:
        r["group"] = group
    return r


def test_split_is_disjoint_complete_and_order_free():
    records = [rec(str(i)) for i in range(2000)]
    dev, test = split.split("d", records, "dev"), split.split("d", records, "test")
    assert {r["id"] for r in dev}.isdisjoint(r["id"] for r in test)
    assert len(dev) + len(test) == len(records)
    assert 0.15 < len(dev) / len(records) < 0.25
    assert [r["id"] for r in split.split("d", records[::-1], "dev")] == [r["id"] for r in dev][::-1]
    assert split.split("d", records, "dev", seed=14) != dev  # seed changes the split


def test_split_keeps_a_group_on_one_side():
    records = [rec(f"{g}:{i}", group=f"t{g}") for g in range(50) for i in range(5)]
    dev_groups = {r["group"] for r in split.split("d", records, "dev")}
    assert dev_groups and dev_groups.isdisjoint(r["group"] for r in split.split("d", records, "test"))
    assert all(sum(r["group"] == g for r in split.split("d", records, "dev")) == 5 for g in dev_groups)


def test_split_rejects_unknown_part():
    try:
        split.split("d", [], "train")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_gretel_en_entity_strings_become_every_occurrence():
    text = "Mail ann@x.io or ann@x.io, SSN 123-45-6789."
    spans = entity_spans(text, [{"entity": "ann@x.io", "types": ["email"]},
                                {"entity": "123-45-6789", "types": ["ssn"]}, {"entity": "", "types": ["name"]}])
    assert spans == [{"start": 5, "end": 13, "label": "email"}, {"start": 17, "end": 25, "label": "email"},
                     {"start": 31, "end": 42, "label": "ssn"}]


def test_argilla_name_parts_merge():
    text = "Dear Ann Lee Berg,"
    gold, _ = normalise_gold(text, [{"start": 5, "end": 8, "label": "FIRSTNAME"},
                                    {"start": 9, "end": 12, "label": "MIDDLENAME"},
                                    {"start": 13, "end": 17, "label": "LASTNAME"}], DATASET_LABELS["argilla_pii"])
    assert gold == [{"start": 5, "end": 17, "label": "PERSON"}]


def test_new_datasets_registered_with_pins_and_publish_rules():
    for name in ("gretel_en", "argilla_pii"):
        module = get(name)
        assert name in MODULES and name in DATASET_LABELS
        assert len(module.REVISION) == 40 and module.SUITE == "pii"
    assert get("argilla_pii").PUBLISH == "scores-only"


def test_entries_are_complete_and_map_to_sct_types():
    assert len(entries.ENTRIES) == 8 and {"sct-default", "sct-gliner"} <= set(entries.ENTRIES)
    for name, entry in entries.ENTRIES.items():
        for key in ("package", "checkpoint", "revision", "config", "threshold", "mask", "device", "gpu_gb"):
            assert key in entry, (name, key)
        assert set(entry["labels"].values()) <= set(entries.SCT_TYPES), name
        assert set(entry["supports"]) <= set(entries.SCT_TYPES), name
        assert entry["gpu_gb"] <= 5, name  # shared box: about 6 GB free per GPU
