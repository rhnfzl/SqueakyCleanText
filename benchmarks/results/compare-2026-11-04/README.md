# SCT vs other PII tools: round 2, 2026-11-04

Round 2 scores SqueakyCleanText 0.8.2 and 17 other offline PII entries on 12 fresh datasets.
SCT was not developed on any of them. The protocol is `benchmarks/compare/PROTOCOL.md`
(round-2 datasets in section 1, contamination in section 17). Every table comes from `TABLES.md`.
You can rebuild it with `python -m benchmarks.compare.report benchmarks/results/compare-2026-11-04`.

## Summary

- **Synthetic data: SCT has the best recall, but not the best precision.** sct-default finds 0.888 of
  masked mentions. Only gliner2-fastino shows no significant difference (diff CI -0.012..+0.009). bardsai-v3 and pii-tracer
  have higher precision and exact-span F1.
- **Real data (Korean, Lithuanian, Russian): SCT leads** among the 6 entries that claim these languages
  (mR 0.807). The sets are small (98 to 300 docs).
- **Only emails, phones and numbers: SCT is 6th.** pii-tracer, bardsai-v3, opf and openmed are ahead.
  PROTOCOL section 4 names this shared view as the headline ranking; this report leads with the full view
  instead (see Deviations).
- **Long English legal docs (RedactionBench): no entry beats SCT on recall** with a significant gap, but
  175 docs give wide CIs. The long-document placement fix leaves 39 unplaced detections for sct-default.
- **Speed:** on 16 CPU threads SCT does 1.55 docs/s at batch 8 and peaks at about 10 GB RAM. Regex tools are far
  faster; most model tools are in the same range on CPU.
- This bundle includes the fix for SCT's unplaced detections (R2-1). SCT was re-run; its recall went up.

## The run

- **Cells.** 18 entries x 12 datasets = 216 cells, 0 failed cells. Each entry saw the same 20,817 docs.
- **Code.** SCT's 11 typed datasets retain b665956 placements. Its RedactionBench cells were re-run on
  102e557 (`code-sha.txt`) to fix long-document placement. The other 14 entries retain 9cea2d0 placements.
  NUS-LT inference ran again for every entry during rescoring because its prediction cache cannot be retained.
  The final scores, counts and sample hashes on all 11 typed datasets match the previous bundle exactly.
  Other entries' adapters and scoring code did not change (`NOTE.md` lists the diffs).
  `code-manifest.sha256` hashes every file of the code copy that ran,
  including `entries.py` and `PROTOCOL.md`. `run.json` has `git_commit: null` because the run used a copied tree
  with no `.git`, so these files are the provenance.
- **Setup.** Default track only (vendor quick start). Test part, up to 300 docs per language. Batch size 1,
  16 threads. Model entries ran on one shared GPU with a 5 GB cap. sct-default, presidio, presidio-full,
  scrubadub, datafog and bardsai-v3 (ONNX) ran on CPU. sct-training was forced to CPU
  (`--config sct-training={"device":"cpu"}`), because its ONNX Runtime CUDA sessions do not respect the cap.
- **Gold-language track.** SCT, presidio and datafog get the true language of each doc.
  SCT's own language detection is not tested here.
- **Entries.** sct-default, sct-training (SCT training preset), sct-gliner, sct-gliner-nvidia
  (SCT with a GLiNER backend), presidio, presidio-full, gliner-nvidia, gliner-knowledgator,
  gliner2-fastino, opf, openmed, bardsai-v3, piiranha, llm-guard, scrubadub, ai4privacy,
  pii-tracer, datafog. Pins are in `PROTOCOL.md` section 2 and `default/envs/`.
- **Groups.** Scored apart and never pooled:
  - r2-privacy-synthetic: newmean, dialogpii, tonic, meddeid, greek_deid, neonredact, pl_pii, europii.
  - r2-privacy-real: kdpii (ko), nus_lt (lt), redmadrobot (ru). CARMEN-I is gated and was not run.
  - r2-untyped: redactionbench (en; gold has no types, so strict F1 is n/a).

## Rules for reading the numbers (decision 21)

- The headline uses clean cells only, and only docs in the languages each entry claims.
- U cells (unknown training data) are shown apart, as information. There are two:
  opf x kdpii, and openmed x kdpii (openmed is built on the OPF base).
- No round-2 cell is excluded (T, E or D). The "excluded sets" column is empty.
- Each headline row uses that entry's own claimed-language sets, so rows cover different set lists.
  sct-default and sct-training claim every language and are scored on all 8 synthetic sets; most others on 5.
  sct-gliner and sct-gliner-nvidia claim English only (their GLiNER model's claim).
  **The main comparison is therefore the pairwise table below:** SCT vs each entry on the same docs,
  datasets clean for both, languages both claim.
- mR = masked recall, mP = masked precision (macro over sets). Doc leak = among docs with at least one
  in-scope gold mention, the share where at least one letter or digit of a gold mention stays unmasked;
  macro over sets, like mR and mP. Doc leak and over-mask: lower is better.
- All totals leave DATE out (PROTOCOL section 4). DATE is still scored: each per-set file keeps a
  `with_date` view.
  `*` means the 95% CI of (entry - SCT) excludes 0. A CI that contains 0 means no significant difference
  was found; it does not show that the scores are equal.

## Main comparison: SCT vs each entry, same docs (synthetic, full view)

From `TABLES.md`, "SCT vs each entry, datasets clean for both, languages both claim". Each row compares the
entry with sct-default on the same docs, so SCT's own value differs per row.

| entry | sets | entry mR | SCT mR | mR diff CI | entry mP | SCT mP | mP diff CI | strict F1 diff CI | doc leak diff CI |
|---|---|---|---|---|---|---|---|---|---|
| gliner2-fastino | 5 | 0.864 | 0.866 | -0.012, +0.009 | 0.640 | 0.739 | -0.113, -0.088 * | +0.043, +0.066 * | -0.150, -0.110 * |
| sct-gliner | 5 | 0.856 | 0.879 | -0.035, -0.012 * | 0.626 | 0.741 | -0.132, -0.101 * | -0.295, -0.263 * | +0.020, +0.067 * |
| bardsai-v3 | 7 | 0.852 | 0.886 | -0.045, -0.024 * | 0.822 | 0.752 | +0.060, +0.080 * | +0.093, +0.124 * | -0.044, -0.003 * |
| sct-gliner-nvidia | 5 | 0.841 | 0.879 | -0.051, -0.024 * | 0.714 | 0.741 | -0.042, -0.013 * | -0.283, -0.244 * | +0.009, +0.056 * |
| sct-training | 8 | 0.838 | 0.888 | -0.054, -0.045 * | 0.816 | 0.750 | +0.056, +0.075 * | -0.000, +0.009 | +0.050, +0.068 * |
| pii-tracer | 8 | 0.837 | 0.888 | -0.070, -0.033 * | 0.781 | 0.750 | +0.015, +0.045 * | +0.063, +0.097 * | -0.101, -0.057 * |
| gliner-knowledgator | 5 | 0.803 | 0.879 | -0.092, -0.057 * | 0.593 | 0.741 | -0.169, -0.130 * | +0.004, +0.041 * | -0.043, +0.022 |
| gliner-nvidia | 5 | 0.760 | 0.879 | -0.140, -0.090 * | 0.681 | 0.741 | -0.078, -0.043 * | -0.048, -0.009 * | -0.096, -0.034 * |
| openmed | 6 | 0.756 | 0.872 | -0.128, -0.104 * | 0.753 | 0.740 | +0.005, +0.020 * | -0.216, -0.189 * | +0.071, +0.112 * |
| presidio | 5 | 0.746 | 0.879 | -0.151, -0.113 * | 0.569 | 0.741 | -0.189, -0.155 * | -0.091, -0.054 * | +0.063, +0.121 * |
| opf | 5 | 0.727 | 0.879 | -0.174, -0.130 * | 0.776 | 0.741 | +0.013, +0.058 * | +0.042, +0.095 * | -0.003, +0.055 |
| presidio-full | 8 | 0.711 | 0.888 | -0.188, -0.167 * | 0.548 | 0.750 | -0.217, -0.186 * | -0.200, -0.174 * | +0.126, +0.165 * |
| scrubadub | 5 | 0.660 | 0.879 | -0.236, -0.201 * | 0.771 | 0.741 | +0.011, +0.049 * | +0.004, +0.042 * | +0.199, +0.253 * |
| llm-guard | 5 | 0.553 | 0.879 | -0.351, -0.300 * | 0.895 | 0.741 | +0.129, +0.175 * | -0.294, -0.248 * | +0.196, +0.255 * |
| piiranha | 5 | 0.488 | 0.863 | -0.390, -0.358 * | 0.829 | 0.739 | +0.074, +0.103 * | -0.353, -0.317 * | +0.308, +0.346 * |
| ai4privacy | 5 | 0.357 | 0.859 | -0.517, -0.487 * | 0.656 | 0.738 | -0.104, -0.060 * | -0.489, -0.456 * | +0.376, +0.414 * |
| datafog | 8 | 0.212 | 0.888 | -0.688, -0.664 * | 0.775 | 0.750 | -0.004, +0.058 | -0.345, -0.311 * | +0.351, +0.390 * |

Sets: 5 = newmean, dialogpii, tonic, meddeid, europii. bardsai-v3 adds greek_deid and pl_pii, openmed adds
neonredact, and the "all languages" entries cover all 8. Diff CIs are entry minus SCT.

## Where SCT wins and where it loses

**Wins (synthetic, full view):**
- Recall. SCT's mR is significantly higher than every other entry except gliner2-fastino (no significant
  difference, diff CI -0.012..+0.009).
- On the 5 sets every entry can be scored on, SCT has the highest mR (0.879; presidio-full 0.877,
  gliner2-fastino 0.872; same-set table, point values).
- Precision vs the GLiNER entries and presidio. SCT's mP (0.74 to 0.75) is significantly above gliner2-fastino,
  gliner-nvidia, gliner-knowledgator, sct-gliner, sct-gliner-nvidia, presidio, presidio-full and ai4privacy.
- Real data: highest mR of the entries that claim ko, lt and ru (see below).

**Losses:**
- Precision vs the precise models. bardsai-v3, pii-tracer, opf, openmed, scrubadub, llm-guard, piiranha and
  sct-training all have significantly higher mP. bardsai-v3 is the strongest all-round entry. On the same docs
  (its languages): mP 0.822 vs 0.752 and strict F1 0.674 vs 0.564, with recall 0.034 lower.
- Exact spans. SCT's strict F1 (0.548) is significantly below gliner2-fastino, bardsai-v3, pii-tracer, opf,
  gliner-knowledgator and scrubadub.
- Docs with a leak. Among synthetic docs that contain in-scope gold mentions, SCT leaves at least one unmasked
  in 0.580 (macro over sets; pooled 0.528). gliner2-fastino (0.437), gliner-nvidia (0.448), pii-tracer (0.502)
  and bardsai-v3 (0.567; SCT 0.590 on the same docs) are significantly lower (each entry's own sets, same macro).
  High recall per mention does not mean fewer leaky docs.
- Shared view (EMAIL, PHONE, NUMBER only). SCT is 6th by point value (mR 0.805). pii-tracer 0.940, bardsai-v3 0.880,
  opf 0.852 and openmed 0.839 are significantly ahead. presidio-full (0.812 vs SCT 0.805 on the same docs, diff CI
  -0.004..+0.019) and presidio (0.795 vs SCT 0.772 on the same docs, diff CI -0.002..+0.051) show no significant
  difference; both point values are above SCT's, and presidio may be ahead. sct-gliner is slightly but
  significantly ahead on the 5 English sets (0.785 vs 0.772).
- sct-training vs sct-default. The training preset trades recall for precision:
  synthetic mR 0.838 vs 0.888, mP 0.816 vs 0.750 (both significant), strict F1 no significant difference
  (diff CI -0.000..+0.009).

## Headline, full view (9 types, DATE excluded from totals), point values

DATE is scored but kept out of every total, as in PROTOCOL section 4; the with-DATE scores are in the
`with_date` view of each per-set file. PROTOCOL section 4 ranks the headline on the shared view; this report
leads with the full view (see Deviations). From `TABLES.md`. CIs, outside-claim and U columns are there. Rows cover different set lists; read them with
the pairwise table above.

| entry | synthetic mR | synthetic mP | synthetic strict F1 | real mR | real mP | untyped mR | untyped mP |
|---|---|---|---|---|---|---|---|
| sct-default | 0.888 | 0.750 | 0.548 | 0.807 | 0.843 | 0.687 | 0.297 |
| gliner2-fastino | 0.864 | 0.640 | 0.642 | - | - | 0.662 † | 0.243 |
| sct-gliner | 0.856 | 0.626 | 0.288 | - | - | 0.681 | 0.213 |
| bardsai-v3 | 0.852 | 0.822 | 0.674 | 0.505 (nus_lt only) | 0.916 | 0.745 | 0.420 |
| sct-gliner-nvidia | 0.841 | 0.714 | 0.304 | - | - | 0.668 † | 0.320 |
| sct-training | 0.838 | 0.816 | 0.553 | 0.724 | 0.883 | 0.662 | 0.343 |
| pii-tracer | 0.837 | 0.781 | 0.629 | 0.461 | 0.787 | 0.332 | 0.720 |
| gliner-knowledgator | 0.803 | 0.593 | 0.593 | - | - | 0.197 † | 0.550 |
| gliner-nvidia | 0.760 | 0.681 | 0.540 | - | - | 0.086 | 0.649 |
| openmed | 0.756 | 0.753 | 0.360 | - (U: 0.613) | - | 0.692 | 0.451 |
| presidio | 0.746 | 0.569 | 0.496 | - | - | 0.692 | 0.262 |
| opf | 0.727 | 0.776 | 0.638 | - | - | 0.351 | 0.519 |
| presidio-full | 0.711 | 0.548 | 0.362 | 0.625 | 0.877 | 0.720 | 0.235 |
| scrubadub | 0.660 | 0.771 | 0.591 | - | - | 0.211 | 0.208 |
| llm-guard | 0.553 | 0.895 | 0.296 | - | - | 0.657 | 0.408 |
| piiranha | 0.488 | 0.829 | 0.252 | - | - | 0.375 | 0.781 |
| ai4privacy | 0.357 | 0.656 | 0.108 | - | - | 0.573 | 0.323 |
| datafog | 0.212 | 0.775 | 0.220 | 0.073 | 0.944 | 0.531 | 0.498 |

`-` = no clean set in a claimed language. † = under 99% complete on that set (see Failures).

## Real data and untyped data (small samples)

**Real data** (kdpii 300 docs, nus_lt 98, redmadrobot 300):
- 7 entries claim ko, lt or ru. 6 have a clean headline row. openmed claims ko, but its only claimed real set
  (KDPII) is a U cell, so it has no headline row (KDPII mR 0.613, information only).
- SCT leads that group: mR 0.807. sct-training 0.724, presidio-full 0.625 and pii-tracer 0.461 are
  significantly lower. bardsai-v3 claims only Lithuanian here; on nus_lt it reaches 0.505 vs SCT 0.934.
- SCT's precision is not the best. datafog (0.944) is significantly higher but finds almost nothing (mR 0.073).
  sct-training (0.883) is also significantly higher.
- Outside their claims, some English-only models come close: gliner2-fastino 0.776, gliner-knowledgator 0.724,
  sct-gliner-nvidia 0.710 (information only).
- Shared view: pii-tracer (0.783) and SCT (0.774) show no significant recall difference (CI -0.058..+0.087,
  too wide to call them equal), and pii-tracer has much higher mP (0.718 vs 0.517).
- 11 of 18 entries claim none of these languages, so most entries have no real-data row at all.

**Untyped** (RedactionBench, 175 long English docs, no types in the gold):
- SCT is 5th of 18 on mR (0.687). bardsai-v3 0.745, presidio-full 0.720, openmed 0.692 and presidio 0.692
  are ahead, but none significantly. CIs are about +-0.13 on this set.
- SCT's mP is low (0.297). bardsai-v3, openmed, datafog, piiranha, opf, pii-tracer, gliner-knowledgator and
  gliner-nvidia have significantly higher mP.
- SCT still has 39 unplaced detections on this set. They count as wrong predictions; no credit is assumed for them.

## The placement fix (R2-1): before and after

SCT cleans text before its NER step: it collapses whitespace and drops lone letters, bracket content and
stray symbols. So a name split over a line break was stored as "Jan Kowalski", while the document still had
the line break. The harness looked for the stored text in the original document, did not find it, and counted
the detection as unplaced: a wrong prediction, with the gold mention scored as leaked. The fix in b665956
(`scripts/benchmark_release.py`) applies the same cleanup to the search, so these detections are placed.
The stored outputs held only placed spans, so the 4 SCT entries were re-run; the other 14 entries kept their
9cea2d0 cells, since nothing they run or are scored with changed.

sct-default, full view, before (9cea2d0) and final (typed sets b665956, RedactionBench 102e557):

| group | mR | mP | strict F1 | doc leak |
|---|---|---|---|---|
| synthetic | 0.858 -> 0.888 | 0.745 -> 0.750 | 0.539 -> 0.548 | 0.604 -> 0.580 |
| real | 0.795 -> 0.807 | 0.844 -> 0.843 | 0.522 -> 0.526 | 0.594 -> 0.587 |
| untyped | 0.643 -> 0.687 | 0.306 -> 0.297 | n/a | 0.854 -> 0.848 |

The fix changed one ranking: bardsai-v3's synthetic recall (0.852) had no significant difference from SCT's
before and is now significantly lower. The precision and strict F1 gaps to bardsai-v3 did not close.

Unplaced detections, before -> after:

| entry | 11 typed sets, original -> final | RedactionBench, original -> b665956 -> 102e557 |
|---|---|---|
| sct-default | 1,247 -> 2 | 572 -> 4,533 -> 39 |
| sct-gliner | 1,174 -> 11 | 343 -> 5,400 -> 39 |
| sct-gliner-nvidia | 1,341 -> 7 | 282 -> 517 -> 79 |
| sct-training | 0 -> 0 | 0 -> 0 -> 0 |

On the 11 typed sets the fix works: 20 unplaced spans are left across the three entries (sct-default dialogpii 2;
sct-gliner newmean 11; sct-gliner-nvidia dialogpii 5, meddeid 1, neonredact 1).

The first fix introduced a long-document regression: literal HTML tags could expand as markers, collapsed
spaces could move context alignment, and shell context could match a distant suffix. Commit 102e557 fixes
these cases and restricts backward recovery to contextual matches. Four SCT RedactionBench cells were run
again, followed by rescoring. The run finished on 2026-11-06 at 02:19 UTC with no failed cells.

These changes mostly affect detections outside the scored label scope, so unplaced counts fall sharply while
headline recall barely changes (sct-default 0.6863 -> 0.6867). Precision falls from 0.308 to 0.297.
The remaining 39 / 39 / 79 unplaced detections count as wrong predictions. Their correct locations and effects
are not established. No other tool has unplaced detections. Counts are in each cell's `unplaced_spans` field.
Synthetic regression tests cover 3,000 documents; they do not establish placement correctness on real typed data.

## Speed

From `speed-16t/` and `speed-gpu/` (`speed.py`, PROTOCOL section 16): 1,000 docs from the FineWeb-2 speed sample,
batch 8 for throughput, 3 repeats, median. CPU: a workstation CPU, cores 0-15, 16 threads, peak RAM from
/proc VmHWM. GPU: one shared GPU with a 5 GB cap per job. The full tables are at the end of `TABLES.md`.

| entry | CPU docs/s | CPU peak RAM | GPU docs/s | GPU peak VRAM |
|---|---|---|---|---|
| datafog | 1,555 | 34 MB | (CPU entry) | - |
| presidio | 26 | 1,080 MB | (CPU entry) | - |
| presidio-full | 13.7 | 19,753 MB | (CPU entry) | - |
| bardsai-v3 | 2.42 | 6,975 MB | (CPU entry, ONNX) | - |
| piiranha | 1.80 | 2,418 MB | 41.8 | 1,884 MB |
| sct-training | 1.56 | 10,205 MB | skipped | - |
| **sct-default** | **1.55** | **10,023 MB** | (CPU entry) | - |
| llm-guard | 1.40 | 2,271 MB | 11.4 | 1,322 MB |
| ai4privacy | 1.37 | 1,534 MB | 20.8 | 1,226 MB |
| pii-tracer | 1.33 | 3,098 MB | 19.8 | 1,826 MB |
| scrubadub | 1.20 | 5,180 MB | (CPU entry) | - |
| gliner2-fastino | 0.63 | 34,356 MB | OOM | - |
| sct-gliner | 0.61 | 6,715 MB | OOM | - |
| openmed | 0.42 (1 of 3 repeats timed out) | 25,637 MB | 9.9 | 5,100 MB |
| opf | timeout | - | 17.0 | 4,626 MB |
| gliner-nvidia | timeout | - | batch-8 OOM | 5,354 MB |
| gliner-knowledgator | timeout | - | OOM | - |
| sct-gliner-nvidia | timeout | - | OOM | - |

- sct-default does 1.55 docs/s (3,534 chars/s) on 16 threads, with an 8.2 s cold start and 581 ms median
  latency. It is mid-pack among model entries on CPU and far behind the regex tools (datafog, presidio).
- Its peak RAM (about 10 GB) is high. Only gliner2-fastino, openmed, presidio-full and sct-training use more.
- "timeout" = all 3 CPU repeats hit the 3,600 s limit.
- GPU errors: gliner-knowledgator, gliner2-fastino, sct-gliner and sct-gliner-nvidia crashed with CUDA
  out-of-memory under the 5 GB cap in every repeat. gliner-nvidia loaded and has latency (24 ms median) but
  ran out of memory at batch 8, so it has no batch-8 docs/s. sct-training was skipped on GPU because its ONNX
  Runtime CUDA sessions do not respect the cap. These are failures under the shared 5 GB cap; no run at a higher
  cap was made.
- sct-default runs on CPU in this protocol, so it has no GPU number.

## Failures in the quality run

Failed docs are scored as unmasked (every mention leaked).

| entry | failed docs | cause | sets under 99% complete |
|---|---|---|---|
| sct-gliner-nvidia | 7 | OutOfMemoryError (redactionbench) | redactionbench 0.960 |
| gliner-knowledgator | 5 | OutOfMemoryError (3 redactionbench, 2 nus_lt) | redactionbench 0.983, nus_lt 0.980 |
| gliner2-fastino | 22 | 2 OutOfMemoryError (redactionbench); 20 ValueError (19 newmean, 1 meddeid) | redactionbench 0.989 |
| scrubadub | 2 | ValueError (newmean) | - |

All OOM failures happened under the 5 GB per-job GPU cap on a GPU shared with other jobs. The run records only
the error type, not the requested size, and there is no control run at a higher cap. So these failures show that
the entries do not fit 5 GB on the longest RedactionBench and NUS-LT docs; they do not show the entries would pass
with more memory. They lower RedactionBench recall by up to 0.014 (sct-gliner-nvidia: 0.682 on completed docs vs
0.668 in the headline). The per-language and per-dataset tables in `TABLES.md` count completed docs only, so for
these entries they read slightly higher than the headline. Every table in `TABLES.md` marks rows and cells that
use a set under 99% complete with † (PROTOCOL section 15).

## Deviations from the plan

- **20% dev part on sets with no own split.** `split.py` holds out 20% as dev on newmean, dialogpii,
  nus_lt, redmadrobot, redactionbench, tonic, greek_deid and pl_pii. SHORTLIST section 3 said to take
  nothing from these sets. Round 2 does no tuning, so the dev part is unused and nothing leaks.
  The cost is test size: RedactionBench 175 of 200 docs, NUS-LT 98 of 123.
- **No tuned track, no recovery audit.** SCT scores include the recovery rule (for example 70 recovered
  copies on sct-default pl_pii). There is no "without recovery" run this round.
- **GPU quality runs ran once,** not twice (PROTOCOL section 14).
- **Three placement versions.** SCT typed-set cells use b665956; SCT RedactionBench cells use 102e557.
  Other entries retain 9cea2d0 placements (adapters unchanged). NUS-LT inference repeats during rescoring.
- **Headline view.** PROTOCOL section 4 ranks the headline on the shared intersection. Round 2's shared view
  is only EMAIL, PHONE and NUMBER, so this report leads with the full view and the pairwise table; this choice
  was made after the results were in. On the shared view SCT is 6th on synthetic data (mR 0.805; pii-tracer
  0.940, bardsai-v3 0.880, opf 0.852, openmed 0.839, presidio-full 0.812). On the full view it is 1st (0.888).

## Limits

- No result here is for SCT's language detection (gold-language track).
- Real data covers only ko, lt and ru, with 98 to 300 docs per set. 11 of 18 entries do not claim
  these languages, and openmed's only claimed set (KDPII, ko) is a U cell.
- RedactionBench is one English set of 175 long docs. Its CIs are wide, and SCT still has unplaced detections there.
- The headline ranks entries over different set lists. Use the pairwise tables for direct claims.
- Speed comes from one shared machine. Other jobs on the GPU can move the GPU numbers.

## What is in this folder

- `README.md`: this report. `NOTE.md`: how this bundle was put together from the three placement versions.
- `TABLES.md`: every table, rendered by `benchmarks.compare.report` from this folder.
- `default/<entry>/<dataset>-test.json`: scores, counts and config per cell. No dataset text,
  no document ids, no per-sample predictions. `sample_id_sha256` is a hash of the sample.
  Run paths are rewritten to repo-relative ones or `<hf-hub-cache>` / `<out>`.
- `default/<entry>/offset-check.json`: the harness's own synthetic offset fixtures.
- `default/run.json`, `default/summary.json`, `default/envs/*.freeze.txt`.
- `speed-16t/speed.json`, `speed-gpu/speed.json`: speed runs, with hardware fields removed.
- `code-sha.txt`, `code-manifest.sha256`: code provenance.
- `MANIFEST.json`: sha256 of every file above except this README.

Not committed: raw predictions and logs. They stay on the run machine. All datasets are scores-only.
NUS-LT prediction caches must be deleted there once scoring is final (RAIL-D licence).
