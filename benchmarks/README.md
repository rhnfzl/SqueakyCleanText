# SCT benchmarks

Download-and-score benchmarks over open datasets. They measure how well SCT finds
and removes PII, how well it detects the language, how it does on long documents,
how well it removes noise, and how fast it runs. This folder is not part of the
PyPI wheel (`pyproject.toml` ships only `sct*`).

The repo holds only download code, the scorer and the scores. Dataset files go to
`benchmarks/.cache/` (gitignored). Result files never hold dataset text.

## Run

```bash
pip install -e . -r benchmarks/requirements.txt
python -m benchmarks.run --suite all --out benchmarks/results/0.8.0/
```

Options:

- `--suite all | pii | long | lid | speed | clean` (several allowed)
- `--datasets openpii tab ...` to run only some datasets
- `--configs default training ner-off` (all three by default)
- `--n 300` docs per language (defaults: pii 300, lid 100, speed and clean 100 per script, long all)
- `--seed 13`, `--threads 4` (ONNX intra-op threads), `--workers 4` (`process_batch` workers)
- `--device cuda --gpu-mem-gb 4` to run NER on a GPU with a memory cap
- `--sct-version` must match the installed `sct`; it only guards against mislabelled results

Gated datasets (REDACT) need a Hugging Face token with access. Set `HF_HUB_CACHE`
to reuse a model cache.

Each run writes `<dataset>.json` per dataset, `run.json` and `summary.md`. Each
dataset file records the run that made it: command, seed, sample size, dataset
revision, SCT version, git commit, package versions, NER models and revisions, CPU,
GPU and ONNX providers. Datasets marked `scores-and-ids` also list the sample ids,
so anyone can rerun the same docs.

On a shared Linux box, pin cores and check GPU memory first:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
taskset -c 0-63 python -m benchmarks.run --suite all --threads 8 --workers 8 --device cuda --gpu-mem-gb 4
```

## Configs

- `default`: `TextCleanerConfig()`.
- `training`: the `training` preset (PII removal and unicode repair only).
- `ner-off`: `default` without NER, to separate regex finds from NER finds.

For scoring, every config also sets `replacement_mode="reversible"` and
`check_to_ascii_unicode=False`: each placeholder then maps back to the exact value it
replaced, which gives the predicted span. Speed runs use the plain configs.

## What is scored

PII (per language and per SCT type):

- **strict**: exact type and span (`sct.evaluation.evaluate_pii`).
- **partial**: a gold span is found when a prediction of the same type overlaps it; a
  prediction is correct when it overlaps a gold span of the same type.
- **masked recall**: type-agnostic partial recall: a gold span counts when any
  prediction overlaps it, whatever its type (a NUMBER masked as PHONE is masked, not
  missed). Shown next to partial recall, overall, per type and per language.
- **leak rate** (the main number): share of gold PII values still in the output, fully
  or as a 4+ character word of the value outside the placeholders. Words are runs of
  letters, digits and combining marks, so Devanagari or Tamil words are not cut at
  vowel signs. Known limit: scripts written without spaces (Thai) give one long word,
  so their partial leaks are undercounted. It is an upper
  bound: the same word elsewhere in the doc also counts. Values are compared after
  NFKC, so a full-width name SCT folds to ASCII still counts. DATE is shown per type
  but is not in the total or the per-language rates.
- **by position**: partial recall by the gold span's start offset (1000-char
  buckets), so recall lost after the first model window shows up.

Gold labels are mapped to SCT types in `labels.py` (a schema mapping, not
detection). Gold types with no SCT type are out of scope: not misses, and a
prediction that only overlaps out-of-scope gold is not a false positive. Adjacent
first name + last name spans (datasets with name-part labels) merge into one PERSON;
two adjacent PER entities in BIO data stay two. SCT `YEAR` counts as `DATE`.
SCT has no ID type: ID numbers map to `NUMBER`, which `default` replaces and
`training` keeps, so `training` leaks them on purpose.

Long docs (TAB) add the TAB masking recall: share of DIRECT and QUASI mentions
whose letters and digits are all inside SCT spans of any type, per annotator, then
averaged over annotators.

Language ID (WiLI-2018): accuracy per language and per script for languages Lingua
knows, the top confusions, and the false-positive rate: the share of paragraphs in
other languages that still get a language.

Speed: docs/s and chars/s per FineWeb-2 language/script after a short warmup, with
the thread and worker counts in the result.

Cleaning: a fixed-seed HTML tag, emoji, URL and mojibake word are put into each
FineWeb-2 doc; the score is the share removed from the output. Emoji removal is off
by default, so `default` keeps emoji on purpose.

## Datasets

| name | suite | source | revision | licence | publish |
|---|---|---|---|---|---|
| openpii | pii | ai4privacy/pii-masking-openpii-1.5m (validation) | a785eb5 | CC-BY-4.0 (Ai4Privacy / Ai Suisse SA) | scores + ids |
| gretel | pii | gretelai/synthetic_pii_finance_multilingual (test) | 7b844d1 | Apache-2.0 | scores + ids |
| nemotron | pii | nvidia/Nemotron-PII (test) | b70ffaf | CC-BY-4.0 | scores + ids |
| redact | pii | guneeshv/REDACT-PII-Benchmark (locked 1,000) | 1d7fef6 | REDACT Dataset Terms v1.0 | scores only |
| iahlt | pii | HebArabNlpProject/arabic-iahlt-NER (test) | f3b43f4 | CC-BY-4.0 | scores + ids |
| bayanat | pii | Bayanat-Labs/arabic-pii-eval (test) | c59d164 | CC-BY-4.0 | scores + ids |
| thainer | pii | pythainlp/thainer-corpus-v2.2 (test) | e516176 | CC-BY-3.0 (HF tag) | scores + ids |
| thai_pii | pii | DuckerMaster/Thai-PII-Synthetic (test) | dcf1b69 | CC-BY-4.0 | scores + ids |
| naamapadam | pii | ai4bharat/naamapadam (hi, bn, ta test) | 9d4f21a | CC0-1.0 per card, CC-BY-NC-4.0 in loader | scores + ids |
| multiconer | pii | MultiCoNER/multiconer_v2 (hi, bn, zh test) | 4be2d62 | CC-BY-4.0 | scores + ids |
| stockmark | pii | stockmark/ner-wikipedia-dataset (all) | d6c4d48 | CC-BY-SA-3.0 | scores + ids |
| gretel_en | pii | gretelai/gretel-pii-masking-en-v1 (test) | e06eb14 | Apache-2.0 | scores + ids |
| argilla_pii | pii | argilla/textcat-tokencat-pii-per-domain (all; 1% of ai4privacy/pii-masking-200k train) | 683954a | none on repo; Ai4Privacy licence upstream | scores only |
| tab | long | NorskRegnesentral/text-anonymization-benchmark (GitHub, test) | 558e09e | MIT | scores + ids |
| wili | lid | MartinThoma/wili_2018 (test) | 12c3a51 | ODbL-1.0 | scores + ids |
| fineweb2 | speed, clean | HuggingFaceFW/fineweb-2 (test splits) | af9c133 | ODC-By-1.0 + CommonCrawl terms | scores + ids |

Notes:

- gretel gold comes from an LLM plus GLiNER, so its precision is a lower bound.
- multiconer entities are public figures and works, not real PII.
- REDACT is gated: aggregate scores only, never text or ids.
- gretel_en gold has no offsets: spans are every exact occurrence of each entity string.
- argilla_pii has no licence of its own and comes from Ai4Privacy data: aggregate scores only.
- Tool comparisons (dev/test splits, protocol): see `compare/PROTOCOL.md`.

Not run yet: bigcode-pii, PIILO and FLORES+ (no access), Wojood (not available),
WikiANN and papluca (unclear licence; secondary), Universal NER (no hi/bn/ta/th),
HPLT v2 (second speed domain; FineWeb-2 covers speed for now).

## Rules

- Never commit dataset files or dataset text, not even small samples.
- `scores-only` datasets publish aggregate scores only.
- Score drops never block a release. Every drop is investigated and reported.
- Never fix a score with a word list (see `AGENTS.md`).

## Add a dataset

1. Add `benchmarks/datasets/<name>.py` with `NAME`, `HF_ID`, `REVISION` (a pinned
   commit), `LICENCE`, `PUBLISH` (`scores-and-ids` or `scores-only`), `SUITE` and
   `load(n, seed)` returning `{"id", "lang", "text", "spans": [{"start", "end", "label"}]}`
   records. Use `hf_file`, `sample_per_lang`, `bio_to_spans` and `valid_spans` from
   `benchmarks.datasets`. Ids must be unique.
2. Add its name to `MODULES` in `benchmarks/datasets/__init__.py` and to a suite in
   `benchmarks/run.py`.
3. For PII data, add its label map to `DATASET_LABELS` in `benchmarks/labels.py`.
4. Smoke-run it: `python -m benchmarks.run --suite pii --datasets <name> --n 3 --out benchmarks/.cache/smoke`.
