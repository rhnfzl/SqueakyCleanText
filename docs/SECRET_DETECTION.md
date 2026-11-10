# Credential format detection

Applications can protect formatted credentials through an optional rule scanner.
This adds a dedicated format scanner alongside existing GLiNER `password`, `api_key`, and `access_token` labels.
The default pipeline and legacy processing stay unchanged.

Install `squeakycleantext[secrets]` to use the pinned `datafog-core==1.0.0` engine.
Register an application-owned factory and select unified processing explicitly:

```python
from sct import TextCleaner, TextCleanerConfig
from sct.detectors import DetectorRegistry, DetectorSpec
from sct.detectors_datafog import DataFogSecretDetector

registry = DetectorRegistry()
registry.register(
    "credentials",
    lambda cfg, options: DataFogSecretDetector(**options),
)
cfg = TextCleanerConfig(
    pii_processing="unified",
    check_ner_process=False,
    detector_specs=(DetectorSpec("credentials", {
        "entities": ["API_KEY", "BEARER_TOKEN", "JWT", "PRIVATE_KEY"],
    }),),
)
cleaner = TextCleaner(cfg, detector_registry=registry)
```

Registration executes trusted application code. Configuration contains JSON data only.
No detector is registered by default. Unknown categories and empty selections fail.
The example selects only credential detection. Include `DetectorSpec("builtin_regex")` explicitly for other format rules.

The adapter calls `datafog_core.scan(text, {"entities": [...]})` without a model session or network access.
It does not use language or caller context and does not advertise context support.
A missing SDK raises `DetectorUnavailable`. A scan failure raises the same value-free exception without exposing the vendor error.
Invalid findings raise a value-free `ValueError`. Neither failure falls back to another detector.

## Findings and policy

Findings refer to the canonical text stage, with zero-based, end-exclusive Unicode code-point offsets.
The 1.0.0 Python binding exposes `Finding.codepoint_range` and `Finding.byte_range` separately.
The adapter uses the former and checks the latter against the corresponding UTF-8 boundaries.
It rejects incomplete, inconsistent, empty, reversed, and out-of-bounds ranges.
It never locates a finding by searching for its matched value.

Public `Detection` values contain canonical types and a finite float score, or `None` when confidence is unavailable.
The adapter copies no matched text, vendor result object, detector attributes, or raw exception message into findings.
Scores from different providers are not calibrated against each other.
Normal unified policy rules apply to all three replacement modes: placeholder, synthetic, and reversible.
Reversible mode intentionally stores protected values in its explicit anonymization map.
Do not log that map or other explicitly sensitive metadata.

## Qualified scope and limits

Synthetic tests exercise four formats: a GitHub-shaped API key, an Authorization Bearer header, a JWT, and a PEM private-key block.
Each positive case repeats the same value twice, with emoji and mixed Unicode before the matches.
Native byte and code-point positions differ and map to the exact Python slice in each case.
Inserting `é漢😀` inside each tested credential produces no finding, consistent with these ASCII credential formats.
These four formats do not establish coverage of every provider or credential syntax.

A six-snippet near-miss cohort contains a short API-key assignment, a short Bearer header, a dotted JWT-like literal,
an incomplete PEM header, an ordinary code string, and a numeric assignment.
The pinned engine reports one false positive: `Authorization: Bearer short`.
It reports no findings for the other five snippets (one false positive in six snippets, 16.67%).
The engine also misses a valid Bearer header when ordinary text precedes `Authorization` on the same line.
The positive Bearer fixture begins on a new line.
These results describe this small synthetic cohort only, not a general false-positive rate or recall estimate.
No lexical suppression or alternate scanner hides these limits. Select the categories your application needs.

Run the required native profile with:

```bash
SCT_OPTIONAL_PROFILE=secrets .venv/bin/python -m pytest tests/test_secret_detector.py -v
```

Core installs run protocol and validation tests without the SDK. Native tests skip only when the optional SDK is absent.
The required secrets profile fails when its declared SDK is missing.
