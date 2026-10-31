# Score columns, data card and sct-report

`sct/score.py` turns each cleaned record into flat, Arrow-friendly `sct_*`
columns. Every value is value-free: counts, shares, type names, warning codes,
scores and hashes. Entity text never appears.

## Score A (always on, free)

Read from `result.signals` (the public view):

| column | type | from |
|---|---|---|
| `sct_language` | `str \| None` | `language` |
| `sct_language_score` | `float \| None` | `language_confidence` |
| `sct_pii_count` | `int` | `pii_count` |
| `sct_pii_share` | `float` | `pii_share` |
| `sct_pii_types` | `list[str]` | sorted keys of `pii_types` |
| `sct_warnings` | `list[str]` | `warnings` |

```python
from sct.score import score_record
row = score_record(cleaner.process(text))
```

## Score B (built, off by default)

Switched on with `ScoreConfig` (in `sct/score.py`, not `TextCleanerConfig`).
When any part of B is on, both `sct_quality` (`float | None`) and
`sct_minhash` (`list[int] | None`) are present so the schema stays fixed.

| field | default | meaning |
|---|---|---|
| `quality` | `False` | run a quality scorer on the ORIGINAL text |
| `quality_model` | `None` | directory written by `scripts/export_quality_head.py` |
| `quality_revision` | `None` | expected head revision; mismatch raises |
| `minhash` | `False` | MinHash of the anonymised text (`pip install 'SqueakyCleanText[score]'`) |
| `minhash_num_perm` | `128` | permutations |
| `minhash_seed` | `1` | seed |
| `minhash_shingle` | `5` | tokens per shingle |
| `minhash_min_tokens` | `5` | below this many non-placeholder tokens the signature is `None` |
| `data_card` | `False` | build a data card |

```python
from sct.score import ScoreConfig, score_records
rows = score_records(results, texts, ScoreConfig(minhash=True), scorer=my_scorer)
```

### Quality

Any callable `texts: list[str] -> list[float]` (`QualityScorer` protocol) works.
It always sees the original text: placeholders such as `<PERSON>` are out of
distribution for quality models, and only SCT has the original in memory.

Documented default: [FineWeb2-HQ-Plus](https://huggingface.co/epfml/FineWeb2-HQ-Plus-Classifier)
(head Apache-2.0, backbone `FacebookAI/xlm-roberta-base` MIT, about 100
languages; licences checked 2026-10-31). Export it once to one ONNX graph
(needs torch and transformers at export time only):

```bash
python scripts/export_quality_head.py --out models/fw2hq-plus
```

`ONNXQualityScorer(dir)` loads `model.onnx` + `tokenizer.json` with
onnxruntime and tokenizers (core deps) and never loads a pickle. Long documents
are truncated to the first 512 tokens.

### MinHash

Computed on the anonymised text, so two documents that differ only in a name,
email or phone get the same signature. Tokens are casefolded; each placeholder
is one token and `<PERSON_3>` (reversible mode) counts as `<PERSON>`. Documents
whose letters are mostly Han, Hiragana, Katakana, Thai, Lao, Khmer or Myanmar
use character tokens, others use word tokens. Synthetic replacement mode
inserts fake values, so it does not normalise; use placeholder or reversible
mode for dedup. Signatures are only comparable when algo, library version,
`num_perm`, seed, shingle size and normaliser match; `sct-report` writes all
of them under `minhash` in the report.

## sct-report

```bash
sct-report data.jsonl --text-column text --preset training \
    [--no-ner] [--quality-model DIR] [--minhash] \
    [--quasi-identifiers city,age_band --k 5] \
    --out report.json [--card card.md] [--scores scores.jsonl]
```

- Input: `.jsonl`, or `.parquet` with `pyarrow` installed.
- `report.json`: input file name, preset, the data card (`summary`: counts,
  language distribution, records per PII type, PII share histogram, warnings,
  NER model provenance, SCT version), plus `quality`, `minhash` and
  `k_anonymity` sections when asked for.
- `--card`: the data card as Markdown.
- `--scores`: one JSON line of `sct_*` columns per input row (no text).
- k-anonymity (`sct.risk.assess_reidentification_risk`) runs only on the
  columns named in `--quasi-identifiers`; no other column is read, and the
  report holds class sizes, never the values.

## Not measured yet

The quality head's CPU speed and how much placeholders would shift its score
are unmeasured. Run the export on the Mac and gpuserver, time fp32 and dynamic
int8 at 512 tokens, batch 32, and record the numbers here. They tune Score B;
they do not change its defaults (off).
