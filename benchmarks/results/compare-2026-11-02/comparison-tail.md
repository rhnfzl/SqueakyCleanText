## Feature matrix (not scored)

| entry | languages (claimed, as scored) | types mapped to SCT's 9 | char spans | reversible | fake values | text cleaning | language ID | batch API | GPU | offline | licence (code / model) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sct-default | all (SCT claims multilingual NER through its WikiNEural fallback; NER models for en, nl, de, es; LID 75 languages) | 9 | via reversible mode | yes (`TokenStore`) | yes (`synthetic` extra) | yes | yes (Lingua) | `process_batch` | optional (CUDA ONNX) | yes | MIT / per NER model card |
| sct-gliner | en (knowledgator/gliner-pii-base-v1.0 card: `language: [en]`) | 9 | via reversible mode | yes | yes | yes | yes | `process_batch` | optional | yes | MIT / Apache-2.0 |
| presidio | en (only en_core_web_lg loaded, both tracks); spaCy packs possible | 8 (no ORG with spaCy) | yes | encrypt / decrypt operator | custom operator | no | no | `BatchAnalyzerEngine` | optional (transformers engine) | yes | MIT / MIT (spaCy model) |
| gliner-nvidia | en | 9 (prompts) | yes | no | no | no | no | library batch | optional | yes | Apache-2.0 / NVIDIA Open Model License |
| gliner2-fastino | en fr es de it pt nl | 9 (prompts) | yes | no | no | no | no | library batch | optional | yes | Apache-2.0 / Apache-2.0 |
| opf | en ("primarily English") | 7 (no ORG, no IP) | yes | no | no | no | no | yes | optional | yes | Apache-2.0 / Apache-2.0 |
| bardsai-v3 | 24 EU | 9 | yes (bundled detector) | no | no | no | no | yes | optional | yes | - / Apache-2.0 |
| datafog | all (regex, no language claim; US shapes, German pack opt-in) | 5 (no PERSON, LOCATION, ORG, URL) | yes | no (hash, redact) | no | no | no | yes | no | yes | MIT / - |

## Failures (rule 19)

Failed docs are scored as unmasked (every mention leaked); the cause for gliner2-fastino is a
`ValueError` from the library's long-doc remap (spans past the text end). No dataset is under 99%
complete. Default track:

COMPLETION_DEFAULT

Unplaced spans (the entry detected something the harness could not place in the source; scored as a
wrong prediction and the gold as missed): only the two SCT entries have them, because SCT returns
rewritten text and its spans are recovered through reversible placeholders (`unplaced_spans` in each
per-dataset file). The recovery rule above is a separate case: copies with no span at all.

## Where SCT stands

There is no clean evidence for or against SCT in round 1, so no win or loss is claimed. The excluded
cells (in brackets in the per-dataset tables) are development signals only, to be measured again on
fresh data: exact spans and types (strict F1) look weak where long digit IDs are typed PHONE
instead of NUMBER; NUMBER/PHONE precision is low (the regexes mask most digit runs, recall-first);
ORGANISATION recall on gretel finance and NER recall on naamapadam (no Indic NER model in SCT) trail
gliner2-fastino. Speed and memory are not a contamination question (FineWeb-2 speed sample): see the
docs/s and peak RAM columns; SCT runs Lingua LID, cleaning and an ensemble of fp32 ONNX NER models
with a session per language.

## Limits of this comparison

- **Round 1 cannot rank SCT.** Its 0.8.1 fixes were developed on 12 of the 14 datasets (D) and the
  other two belong to those datasets' generator families (D-family). Round 2 needs fresh datasets
  that no entry, SCT included, and no generator family of them has seen.
- **The tuned track saw every privacy dev part.** One configuration per entry, chosen on the dev
  parts of all nine privacy sets; the default track is primary.
- **No GPU numbers.** The pinned torch 2.15.0 wheels need CUDA 13.2; the box's driver is 12.6, so
  every entry ran on CPU (gliner-nvidia fp32).
- **Contamination is from cards, not data.** T/E come from model cards; opf and the knowledgator
  model do not disclose training data (U). The Gretel family includes nemotron (NeMo Data Designer,
  Navigator's successor); this changes no cell's status (every entry flagged on one of them is
  already flagged on the others).
- **Claimed languages from cards.** "Primarily English" (opf) is scored as en only. sct-default
  and datafog are held to every language.
- **SCT is scored through its reversible view.** Its spans come from placeholders; unplaced spans
  still count against SCT, and the recovery rule is conservative (any visible trace blocks it).
- **Narrow shared view.** EMAIL, PHONE, NUMBER only; read it with the full view.
- **Gold quality.** gretel finance (LLM + GLiNER), gretel_en (generator strings placed at every
  exact occurrence), argilla_pii (unreviewed suggestions), nemotron (1.5% span mismatches). The
  blind annotation audit (rule 10) was not done, so precision is a lower bound on these sets.
- **Template leakage.** openpii, argilla_pii and ai4privacy-derived sets have no template id: the
  dev/test split is by record.
- **Not run**: the native language track, the common chunker, matched operating points, PR and
  leak-vs-over-masking plots, GPU repeats, downstream utility beyond chrF.
- **One box, one seed.** Bootstrap CIs cover document sampling only.

## Published numbers and scope notes

- OPF's published 96% F1 is per token; its per-span F1 is 0.926 (0.942 on the corrected set).
  Our scores are span or mention level; compare against the span number only.
- Presidio 2.2.366 (latest at the time of the run) is the version scored here.
- NeMo Curator 1.x has no PII stage (the 0.x one was English-only Presidio); it is not an entry.
- Not entries here: **deduce** is GPL-3.0 (checked in its LICENSE file): benchmark use only, no
  code reuse. **StarPII** is gated under custom terms: evaluation only.
