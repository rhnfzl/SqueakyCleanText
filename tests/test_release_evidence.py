"""A release can only reuse complete GitHub results for its own source tree."""

import pytest

from scripts.release_test_evidence import CORE_JOBS, verified_run


COMMIT, HEAD, TREE = "a" * 40, "b" * 40, "c" * 40


@pytest.fixture
def evidence():
    pull = {"number": 7, "merged_at": "2026-11-22", "merge_commit_sha": COMMIT,
            "base": {"ref": "main"}, "head": {"sha": HEAD}}
    run = {"id": 123, "status": "completed", "conclusion": "success", "head_sha": HEAD, "event": "pull_request"}
    jobs = {name: {"name": name, "conclusion": "success", "steps": []} for name in CORE_JOBS}
    marker = {"name": f"Tested PR 7 tree {TREE}", "conclusion": "success"}
    jobs["build (3.12)"]["steps"].append(marker)

    def get(path):
        if path == f"commits/{COMMIT}/pulls?per_page=100":
            return [pull]
        if path == f"actions/workflows/python-package.yml/runs?event=pull_request&head_sha={HEAD}&per_page=1":
            return {"workflow_runs": [run]}
        if path == "actions/runs/123/jobs?per_page=100&filter=latest":
            return {"jobs": list(jobs.values())}
        pytest.fail(f"Unexpected API path {path}")

    return get, {"pull": pull, "run": run, "marker": marker, "worker": jobs["linux-workers (3.14, forkserver)"]}


def test_reuses_complete_results_for_identical_source(evidence):
    get, _ = evidence
    assert verified_run(get, COMMIT, TREE) == 123


@pytest.mark.parametrize(("record", "field", "value"), [
    ("pull", "merged_at", None),
    ("pull", "merge_commit_sha", "d" * 40),
    ("pull", "base", {"ref": "develop"}),
    ("run", "head_sha", "d" * 40),
    ("run", "event", "push"),
    ("run", "conclusion", "failure"),
    ("run", "status", "in_progress"),
    ("worker", "conclusion", "cancelled"),
    ("worker", "name", "unexpected check"),
    ("marker", "conclusion", "skipped"),
    ("marker", "name", f"Tested PR 8 tree {TREE}"),
    ("marker", "name", f"Tested PR 7 tree {'d' * 40}"),
])
def test_refuses_incomplete_or_different_source_evidence(evidence, record, field, value):
    get, records = evidence
    records[record][field] = value
    assert verified_run(get, COMMIT, TREE) is None


@pytest.mark.parametrize(("status", "expected"), [
    ("identical", "verified=true"),
    ("ahead", "verified=true"),
    ("behind", None),
    ("diverged", None),
])
def test_release_requires_commit_on_main(evidence, monkeypatch, tmp_path, status, expected):
    from scripts import release_test_evidence

    get, _ = evidence
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setattr(release_test_evidence.sys, "argv", ["release_test_evidence.py", COMMIT, TREE])
    monkeypatch.setattr(release_test_evidence, "get_github",
                        lambda path: {"status": status} if path == f"compare/{COMMIT}...main?per_page=1" else get(path))

    if expected is None:
        with pytest.raises(SystemExit, match="must be on main"):
            release_test_evidence.main()
        assert not output.exists()
    else:
        release_test_evidence.main()
        assert output.read_text().strip() == expected
