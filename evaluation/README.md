# PII evaluation

This directory holds reviewed ground truth for measuring sensitive-data detection.
Use it to compare models before changing a production default.

`pii_starter.jsonl` is a synthetic smoke fixture. It checks the evaluation path
and common labels. It is not large enough to support accuracy or compliance claims.

## Dataset format

Each JSON Lines record contains one reviewed document:

```json
{
  "id": "support-en-001",
  "text": "Contact Maria Chen at maria.chen@example.com.",
  "language": "en",
  "domain": "support",
  "slices": ["synthetic", "contact"],
  "entities": [
    {"label": "PERSON", "start": 8, "end": 18},
    {"label": "EMAIL", "start": 22, "end": 44}
  ]
}
```

Offsets use Python character positions. `start` is inclusive and `end` is exclusive.
Labels must match the model prediction labels after adapter mapping.

## Prediction format

Prediction files use the same document IDs and entity shape:

```json
{
  "id": "support-en-001",
  "entities": [
    {"label": "PERSON", "start": 8, "end": 18}
  ]
}
```

Every dataset document without predictions counts all reviewed entities as false
negatives. Predictions for unknown document IDs stop evaluation with an error.

## Run an evaluation

```sh
python -m scripts.evaluate_pii \
  --dataset evaluation/pii_starter.jsonl \
  --predictions path/to/model-predictions.jsonl \
  --model model-name-and-revision \
  --output evaluation/results/model-name.json
```

The report contains exact label-and-span metrics overall and by label, language,
domain, and named slice.

## Release rule

Do not change a default model from model-card results alone.

A candidate must meet all of these conditions on a reviewed project dataset:

1. No recall regression for any configured high-risk PII label.
2. No recall regression for any supported language.
3. Acceptable precision for each target domain.
4. Acceptable CPU latency and memory use on the same hardware.
5. A pinned model revision, recorded license, and reproducible artifact.
6. Numerical-equivalence checks for any ONNX export.

Expand the starter corpus with reviewed, representative examples before applying
this rule to a release.

## Candidate benchmarks

The candidate registry records model IDs, immutable revisions, licenses, runtime
requirements, and integration status. It includes Fastino GLiNER2 PII, NVIDIA
GLiNER PII, mmBERT PII, GLiNER2.5, and GLiFormer.

Run a supported candidate directly:

```sh
python -m scripts.benchmark_models \
  --dataset evaluation/pii_starter.jsonl \
  --candidate gliner2-pii \
  --output evaluation/results/gliner2-pii.json
```

The benchmark stores exact-span metrics and document latency. Run the baseline
and candidate on the same machine and reviewed dataset.

`assess_candidate` rejects a candidate when recall drops, F1 gains less than
0.01, p95 latency exceeds 1.5 times baseline, or ONNX parity is not verified.

Published SPY scores place Fastino GLiNER2 PII above NVIDIA GLiNER PII. These
scores do not replace local evaluation because labels, languages, and domains differ.

## Release benchmark

`release_benchmark.jsonl` is a fixed synthetic multilingual set (EN/NL/DE/ES/ZH/JA/HI/BN/TA)
covering names, emails, URLs, international and Indian phone formats, digits glued to
letters, IPv4/IPv6, long CJK and Hindi text, and years/numbers that must not become phones.
Run it once per release, on the same machine, before and after code changes:

```sh
python -m scripts.benchmark_release
```

It writes `evaluation/results/benchmark-<version>.json` and `.md`: exact-span precision and
recall per label and language, PII values still present in the output (full and partial
leaks), cleaned outputs for parity diffs, and default-config speed (docs/s, one warmup
pass, single process). Correctness uses `replacement_mode="reversible"` so each placeholder
maps back to its exact source span, and `check_to_ascii_unicode=False` so leaks stay visible.
A speed gain counts only if recall per label and language does not drop.
