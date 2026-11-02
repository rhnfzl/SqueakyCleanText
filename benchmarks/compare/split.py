"""Dev/test split by document group, for tool comparisons.

A record goes to ``dev`` or ``test`` by a hash of ``seed:dataset:group``, where ``group``
is the record's template or source record (``record["group"]`` when the loader sets one,
else its id). So the split never depends on record order or on other records, and two
records of one template or one source record (REDACT language variants, Nemotron locale
pairs) always land on the same side. ``dev`` is for tuning only; ``test`` is scored once.

    python -m benchmarks.compare.split --datasets gretel_en argilla_pii   # print sizes, no ids
"""

import argparse
import hashlib
import json
from collections import Counter

from benchmarks.datasets import MODULES, get, sample_per_lang

SEED = 13
DEV_SHARE = 0.2
PARTS = ("dev", "test")


def part_of(dataset: str, group: str, seed: int = SEED) -> str:
    """``dev`` for about DEV_SHARE of groups, else ``test``."""
    digest = hashlib.sha256(f"{seed}:{dataset}:{group}".encode()).digest()
    return "dev" if int.from_bytes(digest[:8], "big") < DEV_SHARE * 2**64 else "test"


def split(dataset: str, records: list[dict], part: str, seed: int = SEED) -> list[dict]:
    if part not in PARTS:
        raise ValueError(f"part must be one of {PARTS}, not {part!r}")
    return [r for r in records if part_of(dataset, str(r.get("group", r["id"])), seed) == part]


def load(dataset: str, part: str, n: int | None = None, seed: int = SEED) -> list[dict]:
    """Whole scored split of *dataset*, then *part*, then a seeded *n* docs per language."""
    records = get(dataset).load(None, seed)
    return sample_per_lang(split(dataset, records, part, seed), n, seed)


def sizes(dataset: str, seed: int = SEED) -> dict:
    """Docs, groups and languages per part (no ids, safe for scores-only data)."""
    records = get(dataset).load(None, seed)
    out = {}
    for part in PARTS:
        chosen = split(dataset, records, part, seed)
        out[part] = {"docs": len(chosen), "groups": len({str(r.get("group", r["id"])) for r in chosen}),
                     "langs": dict(sorted(Counter(r["lang"] for r in chosen).items()))}
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", nargs="+", default=list(MODULES))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    for name in args.datasets:
        try:
            print(json.dumps({name: sizes(name, args.seed)}, ensure_ascii=False), flush=True)
        except Exception as error:  # noqa: BLE001  # report gated/missing data and go on
            print(json.dumps({name: {"error": f"{type(error).__name__}: {error}"}}), flush=True)


if __name__ == "__main__":
    main()
