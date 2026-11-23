---
name: verify-stream-lifecycle
description: Use when changing either streaming redactor, its buffer or output queue, or the order of callback and lifecycle changes.
---

# Verify stream lifecycle

A stream returns each accepted piece of text, cleaned, exactly once. A retry that
loses a piece can silently remove content from the caller's document.

Run `.venv/bin/pytest -q tests/test_stream_lifecycle_model.py` from the checkout root.
The required Python package test matrix runs this file through `.venv/bin/pytest tests/`.

## What the model checks

The model tracks accepted input, delivered output, and whether a successful finish
closed the stream. It calls real `feed` and `finish` methods with a regex cleaner.
It explores all pairs of empty retry, append, finish, and failed finish after each
of six starting cases. Then it probes closure. There are 96 traces. The model
compares exact output against fixtures with a known email replacement. It checks
nested callback calls while a window is being processed. It covers these traces
and fixtures. It does not prove token recognition, arbitrary text, thread
schedules, or the rollback of user callbacks.

## Reproduce the failures

Use the same test file with the earlier versions of `sct/streaming.py` described
below. In a scratch checkout, load the historical module into the current test
runtime with the following command.

```sh
legacy=$(mktemp)
git show 'a4a2140^:sct/streaming.py' > "$legacy"
.venv/bin/python -c '
import sys
from pathlib import Path
import pytest
import sct.streaming as streaming
path = Path(sys.argv[1])
exec(compile(path.read_text(), str(path), "exec"), vars(streaming))
raise SystemExit(pytest.main(sys.argv[2:]))
' "$legacy" -q tests/test_stream_lifecycle_model.py -k blocked
```

The `blocked` case loses the cleaned prefix on that revision. Use
`f9eef5f^` and `-k 'failed_feed or split_email'` to see the input loss and
premature closure fixed in that commit. Use `40d684b` and `-k nested` to see
the windowed callback calls enter a stream that is already processing input.

## Keep the model aligned

The model calls `_StreamingRedactor.feed` and `finish` through both public
redactors. Recheck it when `_open_operation`, `_process_window`, or
`_deliver_output` changes. A trace failure can mean the model assigned the
wrong outcome. Read the trace and compare it with the public retry contract in
`docs/ADAPTERS.md` before changing an assertion.

Replay base is `40d684b`. The lifecycle code and model first pass together in
the commit that adds this skill. Advance this record only after replaying the
historical failures and running the model on the new code.
