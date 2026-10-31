"""Fuzz the styled / full-width PII leak class on fixed seeds (``scripts/fuzz_styled_pii.py``).

Every counted failure cluster of the fuzz rounds (research/0.8.0-fuzz-round*.md) is listed with
the first seed that shows it: open ones are xfail(strict=True), fixed ones must stay fixed. A new
cluster on these seeds fails ``test_no_new_failure_cluster``. Move a cluster to FIXED once it passes.
"""

import pytest

from scripts.fuzz_styled_pii import check_case, fold, make_case, render

# Open clusters (research/0.8.0-fuzz-round*.md) -> first seed that still shows them; xfail(strict=True).
KNOWN_CLUSTERS = {}
# Fixed clusters (rounds 0-5) -> first seed that showed them on the old code; they must stay fixed.
FIXED_CLUSTERS = {
    "leak|EMAIL|glue|placeholder": 300345,
    "leak|EMAIL|glue|reversible": 300345,
    "leak|EMAIL|glue|stream": 300345,
    "leak|EMAIL|glue|synthetic": 300345,
    "leak|PHONE|enclosed_pair|placeholder": 48,
    "leak|PHONE|enclosed_pair|reversible": 48,
    "leak|PHONE|enclosed_pair|stream": 48,
    "leak|PHONE|enclosed_pair|synthetic": 48,
    "leak|EMAIL|enclosed_alnum+native_digit+wide_sep|placeholder": 2834,
    "leak|EMAIL|enclosed_alnum+native_digit+wide_sep|reversible": 2834,
    "leak|EMAIL|enclosed_alnum+native_digit+wide_sep|stream": 2834,
    "leak|EMAIL|enclosed_alnum+native_digit+wide_sep|synthetic": 2834,
    "leak|EMAIL|font_sep|placeholder": 72,
    "leak|EMAIL|font_sep|reversible": 72,
    "leak|EMAIL|font_sep|stream": 72,
    "leak|EMAIL|font_sep|synthetic": 72,
    "leak|EMAIL|small_sep|placeholder": 3,
    "leak|EMAIL|small_sep|reversible": 3,
    "leak|EMAIL|small_sep|stream": 3,
    "leak|EMAIL|small_sep|synthetic": 3,
    "leak|EMAIL|wide_sep+zw_join|placeholder": 100177,
    "leak|EMAIL|wide_sep+zw_join|reversible": 100177,
    "leak|EMAIL|wide_sep+zw_join|stream": 100177,
    "leak|EMAIL|wide_sep+zw_join|synthetic": 100177,
    "leak|EMAIL|zw_join|placeholder": 258,
    "leak|EMAIL|zw_join|reversible": 258,
    "leak|EMAIL|zw_join|stream": 258,
    "leak|EMAIL|zw_join|synthetic": 258,
    "leak|EMAIL|zw_inside|placeholder": 9,
    "leak|EMAIL|zw_inside|reversible": 9,
    "leak|EMAIL|zw_inside|stream": 9,
    "leak|EMAIL|zw_inside|synthetic": 9,
    "leak|IP|zw_inside|placeholder": 2,
    "leak|IP|zw_inside|reversible": 2,
    "leak|IP|zw_inside|stream": 2,
    "leak|IP|zw_inside|synthetic": 2,
    "leak|PHONE|zw_inside|placeholder": 19,
    "leak|PHONE|zw_inside|reversible": 19,
    "leak|PHONE|zw_inside|stream": 19,
    "leak|PHONE|zw_inside|synthetic": 19,
    "leak|URL|zw_inside|placeholder": 166,
    "leak|URL|zw_inside|reversible": 166,
    "leak|URL|zw_inside|stream": 166,
    "leak|URL|zw_inside|synthetic": 166,
    "stream_mismatch|sct|PHONE:zw_inside": 1240,
    "stream_mismatch|training|PHONE:zw_inside": 101250,
    "leak|IP|enclosed_alnum+native_digit+wide_sep|placeholder": 2579,
    "leak|IP|enclosed_alnum+native_digit+wide_sep|reversible": 2579,
    "leak|IP|enclosed_alnum+native_digit+wide_sep|stream": 2579,
    "leak|IP|enclosed_alnum+native_digit+wide_sep|synthetic": 2579,
    "leak|IP|enclosed_alnum+native_digit|placeholder": 23,
    "leak|IP|enclosed_alnum+native_digit|reversible": 23,
    "leak|IP|enclosed_alnum+native_digit|stream": 23,
    "leak|IP|enclosed_alnum+native_digit|synthetic": 23,
    "leak|IP|enclosed_alnum+wide_sep|placeholder": 52,
    "leak|IP|enclosed_alnum+wide_sep|reversible": 52,
    "leak|IP|enclosed_alnum+wide_sep|stream": 52,
    "leak|IP|enclosed_alnum+wide_sep|synthetic": 52,
    "leak|IP|enclosed_pair+native_digit+small_sep|placeholder": 301,
    "leak|IP|enclosed_pair+native_digit+small_sep|reversible": 301,
    "leak|IP|enclosed_pair+native_digit+small_sep|stream": 301,
    "leak|IP|enclosed_pair+native_digit+small_sep|synthetic": 301,
    "leak|IP|enclosed_pair+native_digit+wide_sep|placeholder": 110,
    "leak|IP|enclosed_pair+native_digit+wide_sep|reversible": 110,
    "leak|IP|enclosed_pair+native_digit+wide_sep|stream": 110,
    "leak|IP|enclosed_pair+native_digit+wide_sep|synthetic": 110,
    "leak|IP|enclosed_pair+small_sep|stream": 2046,
    "leak|IP|enclosed_pair+wide_sep|placeholder": 76,
    "leak|IP|enclosed_pair+wide_sep|reversible": 76,
    "leak|IP|enclosed_pair+wide_sep|stream": 76,
    "leak|IP|enclosed_pair+wide_sep|synthetic": 76,
    "leak|IP|small_sep|placeholder": 2,
    "leak|IP|small_sep|reversible": 2,
    "leak|IP|small_sep|stream": 2,
    "leak|IP|small_sep|synthetic": 2,
    "leak|PHONE|enclosed_alnum+enclosed_pair+native_digit|placeholder": 2150,
    "leak|PHONE|enclosed_alnum+enclosed_pair+native_digit|reversible": 2150,
    "leak|PHONE|enclosed_alnum+enclosed_pair+native_digit|stream": 2150,
    "leak|PHONE|enclosed_alnum+enclosed_pair+native_digit|synthetic": 2150,
    "leak|PHONE|enclosed_alnum+enclosed_pair+wide_sep|synthetic": 2900,
    "leak|PHONE|enclosed_alnum+native_digit|placeholder": 121,
    "leak|PHONE|enclosed_alnum+native_digit|reversible": 121,
    "leak|PHONE|enclosed_alnum+native_digit|stream": 121,
    "leak|PHONE|enclosed_alnum+native_digit|synthetic": 121,
    "leak|PHONE|enclosed_pair+font_sep+native_digit|placeholder": 212,
    "leak|PHONE|enclosed_pair+font_sep+native_digit|reversible": 212,
    "leak|PHONE|enclosed_pair+font_sep+native_digit|stream": 212,
    "leak|PHONE|enclosed_pair+font_sep+native_digit|synthetic": 212,
    "leak|PHONE|enclosed_pair+glue|stream": 101747,
    "leak|PHONE|enclosed_pair+native_digit+wide_sep|placeholder": 534,
    "leak|PHONE|enclosed_pair+native_digit+wide_sep|reversible": 534,
    "leak|PHONE|enclosed_pair+native_digit+wide_sep|stream": 534,
    "leak|PHONE|enclosed_pair+native_digit+wide_sep|synthetic": 534,
    "leak|PHONE|enclosed_pair+native_digit|placeholder": 242,
    "leak|PHONE|enclosed_pair+native_digit|reversible": 242,
    "leak|PHONE|enclosed_pair+native_digit|stream": 242,
    "leak|PHONE|enclosed_pair+native_digit|synthetic": 242,
    "leak|PHONE|font_sep+glue|stream": 100306,
    "leak|PHONE|font_sep|placeholder": 914,
    "leak|PHONE|font_sep|reversible": 914,
    "leak|PHONE|font_sep|stream": 914,
    "leak|PHONE|font_sep|synthetic": 914,
    "leak|PHONE|glue|placeholder": 413,
    "leak|PHONE|glue|reversible": 413,
    "leak|PHONE|glue|stream": 413,
    "leak|PHONE|glue|synthetic": 413,
    "leak|PHONE|native_digit+wide_sep|placeholder": 963,
    "leak|PHONE|native_digit+wide_sep|reversible": 963,
    "leak|PHONE|native_digit+wide_sep|stream": 963,
    "leak|PHONE|native_digit+wide_sep|synthetic": 963,
    "leak|PHONE|plain|placeholder": 29,
    "leak|PHONE|plain|reversible": 29,
    "leak|PHONE|plain|stream": 29,
    "leak|PHONE|plain|synthetic": 29,
    "leak|PHONE|small_sep+glue|stream": 283,
    "leak|PHONE|small_sep|placeholder": 82,
    "leak|PHONE|small_sep|reversible": 82,
    "leak|PHONE|small_sep|stream": 82,
    "leak|PHONE|small_sep|synthetic": 82,
    "leak|URL|enclosed_pair+wide_sep|stream": 651,
    "leak|URL|glue|placeholder": 5,
    "leak|URL|glue|reversible": 5,
    "leak|URL|glue|stream": 108,
    "leak|URL|glue|synthetic": 5,
    "leak|URL|small_sep|placeholder": 9,
    "leak|URL|small_sep|reversible": 9,
    "leak|URL|small_sep|stream": 9,
    "leak|URL|small_sep|synthetic": 9,
    "leak|URL|wide_sep+zw_boundary|stream": 1623,
    "leak|URL|wide_sep|placeholder": 4,
    "leak|URL|wide_sep|reversible": 4,
    "leak|URL|wide_sep|stream": 4,
    "leak|URL|wide_sep|synthetic": 4,
    "stream_mismatch|sct|EMAIL:glue+PHONE:glue": 100963,
    "stream_mismatch|sct|NUMBER:glue+PHONE:glue": 1364,
    "stream_mismatch|sct|PHONE:glue": 283,
    "stream_mismatch|training|EMAIL:glue+PHONE:glue": 100963,
    "stream_mismatch|training|NUMBER:glue+PHONE:glue": 1364,
    "stream_mismatch|training|PHONE:glue": 283,
}
# Extra seeds (round 3): PHONE|glue leaking again by another root cause, and URLs whose query holds a
# percent-encoded email with a Thai/CJK domain (%40ถมยฌฒ.in), which the round-3 fix must keep whole.
EXTRA_SEEDS = {301478, 301690, 115, 100322, 200040, 300221, 202031, 202264, 300758}
SEEDS = sorted(set(range(10)) | set(KNOWN_CLUSTERS.values()) | set(FIXED_CLUSTERS.values()) | EXTRA_SEEDS)


@pytest.fixture(scope="module")
def clusters():
    found = {}
    for seed in SEEDS:
        for failure in check_case(seed, folds=(True,))["failures"]:
            found.setdefault(failure["signature"], seed)
    return found


def test_rendered_values_fold_back():
    for seed in range(200):
        for value in make_case(seed)["values"]:
            assert fold(render(value)) == fold(value["plain"])


def test_no_new_failure_cluster(clusters):
    assert sorted(set(clusters) - set(KNOWN_CLUSTERS)) == []


@pytest.mark.parametrize("cluster", sorted(FIXED_CLUSTERS))
def test_fixed_cluster(cluster, clusters):
    assert cluster not in clusters


@pytest.mark.parametrize("cluster", [
    pytest.param(cluster, marks=pytest.mark.xfail(strict=True, reason=f"fuzz round 1 open cluster {cluster}"))
    for cluster in sorted(KNOWN_CLUSTERS)
])
def test_known_cluster(cluster, clusters):
    assert cluster not in clusters
