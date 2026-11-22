"""Find reusable GitHub test results for a release checkout."""

import json
import os
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen


CORE_JOBS = {
    "Python static checks",
    "build (3.12)",
    "build (3.13)",
    "build (3.14)",
    "linux-workers (3.13, fork)",
    "linux-workers (3.14, forkserver)",
}


def verified_run(get, commit, tree):
    """Return the Python workflow run that tested this merged source tree."""
    pulls = get(f"commits/{commit}/pulls?per_page=100")
    pr = next((p for p in pulls if p["merged_at"] and p["merge_commit_sha"] == commit
               and p["base"]["ref"] == "main"), None)
    if pr is None:
        return None

    head = pr["head"]["sha"]
    query = urlencode({"event": "pull_request", "head_sha": head, "per_page": 1})
    runs = get(f"actions/workflows/python-package.yml/runs?{query}")["workflow_runs"]
    if not runs:
        return None
    run = runs[0]
    if (run["status"] != "completed" or run["conclusion"] != "success"
            or run["head_sha"] != head or run["event"] != "pull_request"):
        return None

    jobs = get(f"actions/runs/{run['id']}/jobs?per_page=100&filter=latest")["jobs"]
    passed = {job["name"]: job for job in jobs if job["conclusion"] == "success"}
    marker = f"Tested PR {pr['number']} tree {tree}"
    if CORE_JOBS <= passed.keys() and any(
        step["name"] == marker and step["conclusion"] == "success"
        for step in passed["build (3.12)"]["steps"]
    ):
        return run["id"]
    return None


def get_github(path):
    base = f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}"
    request = Request(  # noqa: S310 - fixed HTTPS GitHub API origin
        f"{base}/{path}",
        headers={"Authorization": f"Bearer {os.environ['GH_TOKEN']}", "Accept": "application/vnd.github+json"},
    )
    with urlopen(request, timeout=10) as response:  # noqa: S310 - fixed HTTPS GitHub API origin
        return json.load(response)


def main():
    commit, tree = sys.argv[1:]
    comparison = get_github(f"compare/{commit}...main?per_page=1")
    if comparison["status"] not in {"identical", "ahead"}:
        raise SystemExit("The release checkout must be on main.")

    try:
        run = verified_run(get_github, commit, tree)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Could not verify prior tests ({type(error).__name__}).")
        run = None

    if run is None:
        print("Running the release test suite in this workflow.")
    else:
        print(f"Reusing Python workflow run {run} for tree {tree}.")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        print(f"verified={'true' if run is not None else 'false'}", file=output)


if __name__ == "__main__":
    main()
