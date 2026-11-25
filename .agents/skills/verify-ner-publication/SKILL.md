---
name: verify-ner-publication
description: Use when changing NER model loading, alias reuse, or the tokenizer that sizes inference windows.
---

# Verify NER model publication

A route sends a window to a model only after its tokenizer can size the window.
A request that sees only the pipeline can send a window the model truncates.

Run this check from the repository root. The required Python package test matrix
runs the same tests.

```sh
.venv/bin/pytest -q tests/test_ner_windows.py -k 'publication or parallel_aliases'
```

The model tracks which of two routes are ready. It explores all four two-request
sequences after a failed load, including retries through either alias. The real
pipeline receives the planned windows. A second test calls it from three threads.
These tests cover one shared model, a load failure, and one short tokenizer
limit. They do not exhaust thread schedules or test provider internals.

## Replay the historical failure

Before commit `ec2d743`, `_ensure_loaded` made the pipeline visible before the
tokenizer. Replay that method with its pre-fix body and pause after the pipeline
dictionary write. While it is paused, call `ner_process` for the newly visible
route. The test model accepts eight tokens. The old implementation passed it a
window longer than eight tokens.

The first passing implementation lives in the commit that adds this skill.
The replay uses the test runtime from `d481c46` and the historical writer from
`ec2d743^`. Recheck these results when changing `GeneralNER._ensure_loaded`,
`_model_tokenizers`, `_plan_windows`, or `_LoadedTokenModel`. Update the
verification record after replaying the failing order and running the fixed tests.
