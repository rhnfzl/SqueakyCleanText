# Changelog

## 0.8.0 - 2026-10-31

0.8.0 makes batch cleaning faster, detects all 75 Lingua languages, and adds a value-free
signal view that feeds new framework adapters (Hugging Face `datasets`, DataTrove, LangChain,
Haystack), score columns and data cards (`sct-report`), a reworked MCP server and a
`training` preset. It has many breaking changes; read "Behaviour changes" before upgrading.

### Highlights

- Speed: on the release benchmark (`evaluation/results/comparison-0.7.1-0.8.0.md`)
  `process_batch` on 16 Linux cores is 4.2x faster on short documents (3.69 to 15.55 docs/s)
  and 3.2x faster on long documents (0.65 to 2.08 docs/s); on 64 cores 29.2 docs/s (short).
  Single-document `process` is unchanged within noise (0.95x to 1.10x). On an RTX 4090,
  single documents are about 17x and batches about 4.5x faster than 64 CPU cores. CPU and
  GPU numbers, hardware advice and CUDA setup: `docs/PERFORMANCE.md`.
- Correctness, release benchmark (30 documents, 84 entities, 9 languages): exact-span
  TP/FP/FN 82/2/2 (P = R = F1 = 0.98) and 0 of 67 PII values leaked, identical to 0.7.1.
  `evaluation/results/benchmark-0.8.0.md`.
- New multilingual PII eval (220 synthetic records, 11 languages): the default ONNX
  ensemble scores 0.88 / 0.92 recall / precision overall. GLiNER multi v2.1 (0.45 / 0.68)
  and GLiNER multi PII v1 (0.57 / 0.91) score lower, so the default model is not switched.
  `docs/MULTILINGUAL_EVAL.md`.
- Linux worker processes (`fork`, `forkserver`) are tested in CI; a crash when a forked
  child freed an inherited ONNX session is fixed.

### Added

- `ProcessResult.signals`: a value-free public view (language, confidence, PII counts,
  share and types, regex and entity counts, warnings, model provenance). Never entity text.
  Details under "Behaviour changes".
- Presets: `sct.preset_config("training", **overrides)`, `TextCleanerConfig.from_preset()`,
  `sct.PRESETS`. `training` removes only PII and repairs broken unicode, for training and
  RAG text. Details under "New options".
- Framework adapters (`docs/INTEGRATIONS.md`), each behind its extra:
  `sct.integrations.hf.clean_dataset` / `SCTMap` (`[datasets]`, `Dataset` and
  `IterableDataset`, `score=True` adds score columns), `sct.integrations.datatrove.SCTCleaner`
  (`[datatrove]`, counts `pii`, `pii_<TYPE>` and `sensitive` stats),
  `sct.integrations.langchain.SCTDocumentTransformer` (`[langchain]`) and
  `sct.integrations.haystack.SCTDocumentCleaner` (`[haystack]`, `to_dict` / `from_dict`
  round-trip through `Pipeline.dumps`). Shared options: `columns` (default the main text
  field), `output="overwrite"` (default) or `"new_column"` (`<col>_clean`, raw kept),
  `config` (default the `training` preset; reversible mode is rejected). Document adapters
  write `metadata["sct"]` = signals of the first column. `sct.integrations._runtime`
  (`CleanerHandle`, `clean_columns`, `clean_records`) keeps one cleaner per process, rebuilt
  after a fork and pickled as config only. `docs/RECIPES.md` covers Ray, Spark, Polars,
  DuckDB, dlt and Daft (Polars and DuckDB run by hand).
- Score columns and data cards (`docs/SCORE.md`): `sct.score.score_record` /
  `score_records` give flat `sct_*` columns. Score A (language, language score, PII count,
  share and types, warnings) is always on. Score B is off by default (`ScoreConfig`): a
  quality score on the original text (`ONNXQualityScorer`, any `QualityScorer` callable;
  `scripts/export_quality_head.py` exports FineWeb2-HQ-Plus to one ONNX graph) and a
  PII-normalised MinHash of the cleaned text (`[score]` extra, character shingles for
  Han, Kana, Thai, Lao, Khmer and Myanmar text). `build_data_card` / `render_data_card`.
- `sct-report` command: JSONL or Parquet in; value-free report JSON, optional Markdown data
  card (`--card`), per-row score JSONL (`--scores`), MinHash metadata, and k-anonymity on
  the columns named in `--quasi-identifiers` only.
- MCP server (`docs/MCP.md`): warnings-first output, `outputSchema`, tool annotations,
  `ToolError`s with a remedy, optional file input (`clean_text(path=...)` under
  `SCT_MCP_ROOT`), `SCT_MCP_PRESET`. Details under "Behaviour changes".
- Language: detection over all 75 Lingua languages with a confidence score; any Lingua
  language can be pinned; a language's own `ner_models` entry runs only that model.
- GPU choice: `ner_device`, `cuda_device_id`, `onnx_cuda_provider_options`; every result
  records which models, revisions and execution providers ran (`metadata['provenance']`).
- Multilingual PII eval set: `evaluation/multilingual_pii.jsonl` (en, nl, de, es, zh, ja,
  th, ar, hi, bn, ta; native-script names, a code-switch slice), built deterministically by
  `scripts/build_multilingual_eval.py` and scored by `scripts/run_multilingual_eval.py`.
- New extras: `datasets`, `datatrove`, `langchain`, `haystack`, `score`.
- CI: a required `linux-workers` job (Python 3.13 `fork`, 3.14 `forkserver`) runs the
  worker-process tests and the adapter tests with their own coverage gate; lint covers
  `scripts/`.

### Behaviour changes

- Dates: `check_replace_dates` (legacy global `CHECK_REPLACE_DATES`) now defaults to `True`, so a full date becomes one `<DATE>` before the year and number steps: `2023-10-01` was `<YEAR><NUMBER><NUMBER>`, now `<DATE>`, in placeholder, synthetic and reversible mode, counted once as `DATE` in `regex_counts`. The numeric pattern now covers ISO 8601 and day/month/year with one consistent `-`, `/` or `.` separator in either order (`2024/1/15`, `2024.01.15`, `15.01.2024`, `01/15/2024`), with an optional time (`T` or space, seconds, fraction, `Z` or offset). Month and day must be in range (year-last takes day/month or month/day, so `31/31/2024` and `14.25.2024` are not dates), year-first with dots needs two-digit month and day, and a digit joined by `.` or `,` on either side blocks a match, so versions (`3.12.0`, `10.1.12.2024`, CalVer `2024.1.5`), IPs, decimals and phone numbers are not dates. `check_replace_dates=False` restores the 0.7.1 output. The `training` preset keeps dates as written.
- ASCII folding: `check_to_ascii_unicode` (legacy global `CHECK_TO_ASCII_UNICODE`) now defaults to `False`, so the library, `sct-clean` and `sct-mcp` keep every script: `日本語` stays `日本語` (was `Ri Ben Yu`), `München` stays `München`. On the release benchmark the default NER ensemble found 0 of 8 Hindi, Bengali and Tamil names with folding on (they came out as transliterated text) and over-tagged folded Japanese (69 tags for 3 names); with folding off it matches the gold count in every language. Folding also broke IDN URLs (`https://例え.jp/パス` became `https://Li e.jp/pasu`, not replaced). `check_to_ascii_unicode=True`, `sct-clean --to-ascii` or `SCT_MCP_TO_ASCII=1` restore 0.7 output. IPv4 addresses written in other digit scripts (`١٩٢.١٦٨.١.١`, `८.८.८.८`) are now replaced without folding (they leaked as `<NUMBER>.١.١`).
- Python 3.11 is no longer supported; 0.8.0 needs Python 3.12 or newer. pip on Python 3.11 keeps resolving to 0.7.1.
- `lingua-language-detector` minimum raised from 2.1.0 to 2.2. Lingua 2.2 can detect the language of some texts differently from 2.1.1, which changes the language-specific NER model, stopwords and date handling chosen for those texts.
- ONNX NER models no longer share one lock per model: threads in `process_batch` now run ONNX inference at the same time (`InferenceSession.run` is thread-safe). Output is unchanged. Torch models and GLiNER still run one call at a time. With the ONNX Runtime default threads, a session uses every physical core of the machine and ignores `taskset`/CPU affinity; set `onnx_intra_op_threads` to cap CPU use. On the benchmark, fewer threads per session was slower, not faster, because the batched NER pass runs a few large calls rather than one per worker.
- Two language keys that use the same model repo with different revisions now load each revision. Before, the second key silently reused the first key's revision. The INT8 quantized cache directory is now keyed by the resolved Hub commit (`<model>@<digest>`, a lowercase hash of the commit), so a quantized file is only reused for the exact weights it came from, also when a branch such as `main` moves. The 0.7.1 directory (`org__model`) is not reused, so each quantized model is quantized again once.
- `ner_batch_size` (default 8, new legacy module global `NER_BATCH_SIZE`) now batches model calls: NER chunks of one document, and of all documents in one `process_batch` call, go through each ONNX, torch or GLiNER model `ner_batch_size` chunks per forward pass (rows right-padded). `process_batch` runs the steps before NER per document in threads, then one batched NER pass over all documents (ONNX batches run on the same number of threads), then the steps after NER. Output text, languages and finding spans are unchanged; padded batches can move scores in the last digits (max 5e-6 on the release and long-document sets with the default models). `ner_batch_size=1` keeps the 0.7.1 call pattern and scores exactly. GLiNER2 still runs one chunk per call.
- Throughput with the default models on a 16-core CPU (`scripts/benchmark_release.py --batch 64`, docs/s, `ner_batch_size` 1 vs 8): short documents 14.8 vs 14.9; long documents (about 8 joined documents each) 2.66 vs 2.38 with cpu-count workers, 1.50 vs 1.64 with one worker. With many workers on CPU, threads already fill the cores, so batching mainly adds padding and memory (peak RSS on the long set about 6 GB vs 12 GB). The release benchmark on another machine (Threadripper, 16 cores) measured the opposite: `ner_batch_size` 8 vs 1 was 15.55 vs 10.31 docs/s (short) and 2.08 vs 1.82 (long) with 16 workers. Which setting is faster depends on the CPU; measure before changing it. `ner_batch_size=1` always uses less memory.
- With `ner_batch_size > 1` and NER on, `metrics_callback` `duration_ms` includes the time a document waits for the shared NER pass of its batch.
- NER now collects the spans of all chunks of a document in document coordinates, votes once and replaces once per document, instead of voting and replacing chunk by chunk. Chunks and model input are unchanged. Differences, only for documents longer than one chunk:
  - `replacement_mode='reversible'`: placeholder numbers (`<PERSON_0>`...) are assigned right to left over the whole document, as for short documents, so indices can differ. Deanonymisation still round-trips.
  - `replacement_mode='synthetic'`: Faker values are drawn in whole-document order, so the fake values can differ.
  - Placeholder mode: two entities of the same type separated only by spaces at a chunk boundary now merge into one placeholder (`<PERSON>` instead of `<PERSON> <PERSON>`), as they already did inside a chunk. If any chunk has a label outside PER/LOC/ORG/MISC, the whole document uses the plain right-to-left replacement that short documents with such labels use.
  - Privacy-policy output and findings are unchanged.
- `ProcessResult.metadata` is now always a dict (it was `None` when no extended feature was on). It always holds two value-free count maps: `regex_counts` (every regex replacement per type: `URL`, `HTML`, `EMAIL`, `IP`, `DATE`, `YEAR`, `PHONE`, `NUMBER`, `CURRENCY`, each `{"count", "chars"}` with `chars` = length of the replaced source text) and `entity_counts` (NER findings per entity type, same shape). Counts cover placeholder, synthetic and reversible mode and fuzzy dates; marker text already in the source and IPs kept by `replace_ips_public_only` are not counted. With `replace_with_currency_symbols=None` each symbol is one count. Entity text stays out of metadata unless `include_entities=True` or a privacy policy is set.
- New `ProcessResult.signals` (`sct/signals.py`: `public_view`, `merge_views`, `PUBLIC_VIEW_KEYS`, `WARNING_CODES`): a value-free view with the same keys for every input, including empty text and `process_json` (language, confidence and source, `chars`, PII count, chars, share and types, regex and entity counts, `sensitive`, warnings, model provenance). Metadata gains `source_chars`; empty or whitespace input now gets full metadata with the `empty_input` warning instead of `None`. `StructuredResult.signals` merges every string leaf; `process_rag_chunks` puts `metadata["sct"]`. Score columns, `sct-report`, the framework adapters and the MCP server read it.
- `repr(ProcessResult)` shows only `lm_text`, `language` and `signals`; `repr` of `DetectedEntity`, `MapEntry` and `AnonymizationMap` no longer shows entity text or originals.
- `sct.adapters.is_sensitive_result(result)` now takes only the result and reads these counts (any regex replacement or NER finding). The old `(cleaner, source, result)` signature and the marker-count guessing are gone; it was wrong for empty or custom markers (a phone replaced with `""` was not sensitive). `process_json`, `process_rag_chunks` and OCR region selection use the new counts.
- Language detection now covers all 75 Lingua languages (high accuracy, about 1.4 ms per short document and 246 MB peak for the detector). Text in Hindi, Thai, Arabic, Chinese, Japanese, Bengali, Tamil and other languages is labelled with its own language instead of English/Dutch/German/Spanish or `None`. EN/NL/DE/ES labels are unchanged on a 41-sentence regression set, and release-benchmark outputs and scores are identical (its zh/ja/hi/bn/ta documents were `None` and ran the same MULTILINGUAL + ENGLISH models). Texts in other languages (French, Italian, Portuguese, Polish, ...) that were mislabelled as one of the four now get their own label, so they run the MULTILINGUAL + ENGLISH NER fallback and English stopwords instead of the mislabelled language's. Very short or ambiguous texts can get a near language (for example `"xyz"` is now POLISH with low confidence). `detect_all_languages=False` restores the 0.7 detector.
- `language=` accepts any Lingua language as a pin or in a tuple (`language='hi'` raised `ValueError` before unless the language was also in `extra_languages`/`ner_models`/...). `supported_languages` now only means "has stopwords, dates or an NER key".
- A language with its own entry in `ner_models` / `torch_ner_models` (for example `{"HINDI": "your/model"}`) and no `ner_ensemble` route now runs only that model; before, it ran MULTILINGUAL + ENGLISH. The built-in FRENCH/PORTUGUESE/ITALIAN keys, which point at the default multilingual model, keep the fallback.
- `ProcessResult.metadata` also always holds `language_confidence` (Lingua top-1 confidence; `1.0` when pinned; `None` when unknown), `language_source` (`pinned`, `detected`, `fallback` when a `language` tuple fell back to its first entry, `none`) and `warnings` (sorted codes `language_unknown`, `language_fallback`, `language_low_confidence`, `ner_fallback_models`).
- `ner_confidence_threshold` and `gliner_threshold` now default to `None` ("not set") instead of `0.85` / `0.4`; legacy `NER_CONFIDENCE_THRESHOLD` is `None` too. Before, PII mode replaced any value equal to the default, so an explicit `ner_confidence_threshold=0.85` or `gliner_threshold=0.4` was silently turned into 0.3. Now an explicit value is always used. New read-only properties give the values that run: `effective_gliner_threshold` (explicit, else 0.3 in PII mode, else 0.4) and `effective_ner_confidence_threshold` (explicit, else the GLiNER threshold in PII mode, else 0.85). Code that read `cfg.ner_confidence_threshold` as a number must use the property. Explicit values outside 0..1 raise `ValueError`. `ner_backend` and `gliner_labels` keep the old "equals the default" rule in PII mode (an explicit `gliner_labels=('person', 'organization', 'location')` still becomes the PII labels).
- `metadata['detector']` (only set with `include_entities=True` or a privacy policy) is gone. Every result now has `metadata['provenance'] = {"backend", "models"}`: one `{"key", "id", "revision", "provider"}` entry per NER model that runs for the document's language, with the resolved Hub commit as `revision` for ONNX models and the execution provider (or torch device) that actually ran it. `{"backend": None, "models": []}` when NER is off.
- `metadata['warnings']` gets `cuda_fallback` when CUDA was requested but a model runs on CPU.
- `ner_device=None` (auto) keeps GLiClass on CPU as before; only `ner_device='cuda'` moves it to the GPU.
- The URL regex no longer takes `"`, `<` or `>` into the path (they are never raw in a URL). `<a href="https://x.org/a">docs</a>` now becomes `<a href="<URL>">docs</a>` instead of `<a href="<URL> ...` eating the rest of the tag; this only shows when HTML is kept (`check_replace_html=False`).
- URL regex: user info (`user:pw@`) stops at `/ ? # " < > @`, so `https://x.org/?to=alice@example.com&cc=bob%40example.net` is one `<URL>` (the percent-encoded email leaked before) and `<a href="https://x.org">alice@example.com</a>` keeps its markup. A bare host with a query or fragment (`x.org/?q=1`, `host#frag`) is replaced whole; a bare trailing `?` stays punctuation.
- Phone regex: grouped national numbers with a trunk 0 (`06 12 34 56 78`, `010-6552 1234`, `020 7946 0958`, 9 or more digits, optional extension) and 3-3-3 groups (`699 123 456`) are now replaced whole; before they were missed or cut. `00` stays the international prefix. Number tables with such groups can now become `<PHONE>`.
- Email regex: the local part accepts RFC punctuation after its first character (`! # $ % & ' * + -`) and single dots, and cannot end with a dot, so `o'brien@x.org` and `a!b@x.org` are replaced whole. `= / | { } ~ ^ ?` and backtick stay out, so `key=alice@x.org` keeps `key=`.
- Marker-shaped text already in the input (`<LONDON>`, also after unicode repair: `＜LONDON＞`, `&lt;LONDON&gt;`) is no longer protected from NER, and reversible mode never assigns a placeholder spelled like one already in the input. Only SCT's own placeholders are protected.
- `metadata['entity_counts']` counts overlapping findings once (merged under the label with the highest score, the label that is replaced).
- `WindowedStreamingRedactor` never splits an email, phone, URL or IP between windows (matched on an NFKC view, so fullwidth digits count), and holds back a phone-like digit run at the end of a feed until the next feed completes it. If no safe split point is left, it raises `BufferError`.
- `process_dataframe` reads cells by position, so frames with duplicate row or column labels are cleaned (before, those cells were skipped).
- Reversible mode on empty input returns an empty `AnonymizationMap` (or stores one in the token store); `sct-clean --token-map` writes it.
- `sct.risk.assess_reidentification_risk` groups on `(type, value)`, so `True` and `1`, or `1` and `1.0`, are different quasi-identifier values. `sct-report` JSON-encodes values the same way (`[1]` and `"[1]"` differ).
- MCP `clean_text` / `clean_json` return `{warnings, summary, text|value, signals}` in that order; `detail="full"` adds value-free `findings` (`entity_type, score, start, end, action, reason`, plus `path` for JSON). `stat_text`, `language`, `policy` and the nested `findings[].entity` / `findings[].finding` shape are gone. Both tools declare `outputSchema` and read-only, non-destructive, idempotent, closed-world annotations. Errors are `fastmcp.exceptions.ToolError` with a remedy (a PII-mode mismatch raised `ValueError`). `create_mcp_server(cleaner=None, *, root=None, max_bytes=None, preset=None)`; concurrent first calls build one cleaner.
- GLiNER on CUDA: after an ONNX-to-torch fallback, and for GLiNER2, the model now moves to the chosen GPU; `presidio_gliner` passes the torch device (`map_location`) and reports it in provenance.

### New options

- Presets: `sct.preset_config(name="training", **overrides)`, `TextCleanerConfig.from_preset(...)` and `sct.PRESETS`. `"training"` only removes PII and repairs broken unicode, for training and RAG text: URL, email, phone, IP replacement, NER (default ONNX ensemble), ftfy repair and language detection on; ASCII transliteration, numbers, years, dates, fuzzy dates, isolated letters and symbols, HTML, backslash-escape decoding, bracket and brace removal, currency, whitespace normalisation, casefolding, stopwords, punctuation and `stat_text` off, so line breaks, indentation and code are kept. `"sct"` is the library default config. Any field overrides; an unknown name raises `ValueError`. GLiNER PII stays opt-in (`ner_mode="pii"`). Known limit: a URL with a path swallows a trailing sentence period (`see https://x.org/a.` becomes `see <URL>`).
- `check_decode_unicode_escapes` (default `True`, legacy module global `CHECK_DECODE_UNICODE_ESCAPES`): part of `check_fix_bad_unicode`; decodes literal backslash escapes (`\u00e9`, `\n`) before ftfy. `False` keeps them as written, for code and JSON.
- `detect_all_languages` (default `True`, legacy module global `DETECT_ALL_LANGUAGES`): auto-detect over all 75 Lingua languages; `False` = 0.7 detector over `supported_languages`. To restrict detection, pass a `language` tuple.
- `language_margin_warning` (default `0.15`, legacy module global `LANGUAGE_MARGIN_WARNING`, 0..1) and `language_confidence_warning` (default `0.0` = off, legacy module global `LANGUAGE_CONFIDENCE_WARNING`, 0..1): `metadata['warnings']` gets `language_low_confidence` when the detected language leads the runner-up by less than the margin, or its confidence is below the confidence threshold. Never changes routing. Measured on the multilingual eval set and release benchmark (250 texts, plus 10/20/40-character prefixes, 994 detections, 83 wrong): the earlier default (confidence below `0.5`) fired on 272 and caught 74 wrong; the margin default fires on 178 and catches 73. On full texts it fires on 11 of 250 instead of 20 and catches all 4 wrong detections instead of 3 (it also flags a Hindi text detected as Marathi at confidence 0.57). Most of the old false alarms were correct English with names. `language_confidence_warning=0.5, language_margin_warning=0.0` restores the earlier warning.
- `fix_character_width` (default `False`, legacy module global `FIX_CHARACTER_WIDTH`): part of `check_fix_bad_unicode`. ftfy's width fix turned the Chinese comma `，` into `,` while `。` stayed, mixing punctuation in CJK text, and widened half-width katakana. The default now keeps CJK punctuation and katakana width and converts only full-width ASCII letters and digits (plus `＠ ． － ＿ ＋` and the ideographic space inside them), so full-width phones and emails are still detected and replaced. `fix_character_width=True` restores the 0.7 behaviour.
- `ner_subword_widen` (default `1.0`, legacy module global `NER_SUBWORD_WIDEN`, `None` or >= 0): WordPiece models widen an entity to its whole word only when the added text is at most this many times the tagged span. Thai and Japanese kana have no spaces, so a whole clause is one WordPiece word: `นครราชสีมา` became `ย้ายไปนครราชสีมาเมื่อปีที่แล้ว...` as one LOCATION. `None` = always widen (the earlier 0.8.0 fix), `0` = never (0.7). Multilingual eval (`sct-default`): Thai recall 0.65 to 0.85.
- `ner_subword_widen_same_script` (default `True`, legacy module global `NER_SUBWORD_WIDEN_SAME_SCRIPT`): WordPiece widening never adds a piece whose letters are in another Unicode script than the entity's edge letter. mBERT reads `はDavid Millerで` and `ถึงคุณSarah Lee` as one word, so the Latin name took the kana or Thai around it. Multilingual eval (`sct-default`): 573/56/87 to 578/51/82 (recall 0.868 to 0.876, precision 0.911 to 0.919), code-switch slice 0.78/0.82 to 0.83/0.88, ja 0.63/0.68 to 0.70/0.75, th 0.85/0.89 to 0.87/0.91, PERSON 0.95/0.94 to 0.97/0.96; the 5 changed documents only lose kana/Thai from a PERSON span. Release benchmark outputs identical, leaks 0/67. `False` restores widening across scripts.
- `legacy_type_names` (default `False`, legacy module global `LEGACY_TYPE_NAMES`): one name per PII type on every surface. Before, phone and IP were `<PHONE>` / `<IP>` as tags but `PHONE_NUMBER` / `IP_ADDRESS` in reversible placeholders (`<PHONE_NUMBER_0>`), `regex_counts`, `signals['pii_types']`, score columns, DataTrove `pii_<TYPE>` stats, MCP summaries and PII-mode GLiNER tags (`phone_number` gave `<PHONE_NUMBER>`). Now all of them say `PHONE` / `IP`, like `EMAIL`, `URL`, `DATE`, `YEAR`, `NUMBER` and the NER types `PERSON`, `LOCATION`, `ORGANISATION`, `MISC`, which already had one name. The short names were kept because they are what the default tag text already showed, so default placeholder output does not change. `PII_LABEL_MAP` maps `phone_number` -> `PHONE` and `ip_address` -> `IP`; PII mode now expands `positional_tags` from the configured `gliner_label_map` (a custom target such as `TEL` was filtered out before). Privacy-policy rules key on the new names. The eval datasets (`evaluation/*.jsonl`) label phones `PHONE`; the release benchmark still maps the old names. `legacy_type_names=True` restores the 0.7 names; tag text stays set by `replace_with_*`.
- `fold_styled_characters` (default `True`, legacy module global `FOLD_STYLED_CHARACTERS`): part of `check_fix_bad_unicode`. With the new default `check_to_ascii_unicode=False`, styled PII leaked: `john@𝐞𝐱𝐚𝐦𝐩𝐥𝐞.com`, `ⓙⓞⓗⓝ@example.com`, `https://𝐞𝐱𝐚𝐦𝐩𝐥𝐞.com/a` stayed in the output and `ｈｔｔｐｓ：／／ｅｘａｍｐｌｅ．ｃｏｍ／ａ` was only half replaced. Letters and digits whose Unicode decomposition is `<font>` or `<wide>` now fold to their NFKC ASCII form, `<circle>` ones too when they touch ASCII letters/digits or URL punctuation, and full-width URL/email punctuation folds between ASCII letters/digits; `＠` also folds between letters of any script, `．` beside an ASCII letter and a `：／／` scheme after ASCII letters, so `山田＠example.com`, `taro＠例え.jp` and `https：／／例え.jp/パス` are replaced; inside an address-shaped run (word characters around an `@`) full-width `．－＿＋` fold beside any script, so `john@例え．テスト` and `josé＿smith@example.com` are replaced whole while `東京－大阪` in prose keeps its form (also with `fold_styled_characters=False`). CJK punctuation in prose, a lone `①`, super/subscripts and squared units keep their form. Release benchmark and multilingual eval unchanged (eval texts normalise byte-identically). `False` restores the earlier 0.8.0 behaviour.
- `ner_device` (default `None` = auto, CUDA when available; `'cpu'`, `'cuda'`), `cuda_device_id` (default `0`) and `onnx_cuda_provider_options` (default `None`, extra `CUDAExecutionProvider` options such as `{"gpu_mem_limit": 8 << 30}`, passed to ONNX Runtime unchecked). Legacy module globals `NER_DEVICE`, `CUDA_DEVICE_ID`, `ONNX_CUDA_PROVIDER_OPTIONS`. Before, every model ran on GPU 0. GLiNER in ONNX mode (`gliner_onnx=True`) keeps its own session setup.
- `json_key_cleaning` (default `'pii'`, legacy module global `JSON_KEY_CLEANING`): how `process_json` (and `sct-clean --json-input`, the MCP `clean_json` tool, RAG chunk metadata) cleans dict keys. `'pii'` runs only the regex PII steps on keys (emails, phones, URLs...; no NER, because models tag lone words such as `message` or `user` out of context, which turned common keys into `<MISC>` and two such keys into a duplicate-key error; the steps the `training` preset turns off stay off), so with the library default `{"k": 1}` keeps its key instead of becoming `{"": 1}`, `{"b": 1, "c": 2}` no longer raises "duplicate key", and keys are not ASCII-folded. `'pii_ner'` adds NER on keys; `'full'` cleans keys with the whole config (0.7 behaviour); `sct-clean --json-input` reports a duplicate key as a usage error instead of a traceback; `'off'` keeps keys as written (keys can then carry PII). Values are unchanged.
- `synthetic_seed` (default `42`, legacy module global `SYNTHETIC_SEED`) and `synthetic_per_document` (default `True`, legacy module global `SYNTHETIC_PER_DOCUMENT`): synthetic mode seeds each document's fakes from the base seed, a hash of the document text and the fake's position, so documents get different fake names and emails (before, every worker thread restarted one seeded stream, so a `process_batch` gave every document the same first fake per type, and sequential output depended on call order), the same document always gets the same fakes in `process`, `process_batch` (any workers, batched NER or not), a new cleaner or an adapter worker process, and repeated values inside one document keep one fake. `False` keeps the 0.7 single stream per thread.
- `synthetic_locale` (default `None`, legacy module global `SYNTHETIC_LOCALE`): synthetic fakes follow the detected document language. Before, every fake was US English (`Faker()`), so Dutch text got `John Smith` and `New York`. The Lingua language maps to a Faker locale through Python's `locale.normalize` territory table, then Faker's bare or first locale of that language (DUTCH -> `nl_NL`, GERMAN -> `de_DE`, CHINESE -> `zh_CN`, JAPANESE -> `ja_JP`, ARABIC -> `ar_AA`, HINDI -> `hi_IN`); 53 of Lingua's 75 languages (English included) get a native locale, the rest (and no detected language) use `en_US`. Faker locales without a provider fall back to its `en_US` one. Fakes stay deterministic per document across threads and processes, also for locale providers that draw from Python's global `random` (th_TH, uk_UA ID numbers, uk_UA plates) or order cities by hash seed (it_IT). A string forces one locale; `'en_US'` restores the 0.7 fakes.
- `ner_sees_tags` (default `True`, legacy module global `NER_SEES_TAGS`): NER reads the same text in every `replacement_mode`. Before, synthetic mode put the fake email, phone and URL in before NER and reversible mode put `<EMAIL_0>`, so NER read a different sentence than in placeholder mode and found different entities (with the small test model, `Jan de Vries werkt bij Philips in Eindhoven, mail jan@vries.nl.` got an ORGANISATION only in synthetic mode; reversible offsets after a regex value were shifted). Now the regex steps put the placeholder-mode tag in the text NER reads, and each regex value goes back in after NER, in order (same fakes as before, per-document mapping and `regex_counts` unchanged; a tag already in the source stays). Multilingual eval (`sct-default`, reversible mode): 578/51/82 to 584/50/76 (recall 0.876 to 0.885, precision 0.919 to 0.921), LOCATION 0.97/0.90 to 0.99/0.93, ORGANISATION 0.66/0.89 to 0.69/0.87, bn recall 0.78 to 0.83, ja 0.70/0.75 to 0.73/0.77, ta 0.87/0.88 to 0.88/0.90, ar precision 0.80 to 0.79; release benchmark outputs identical, leaks 0/67. A regex value inside an NER span goes back into that span's reversible map entry (`University of ann@example.com` as one ORGANISATION restores the email, also when the span cuts the tag, as in `of <EMAIL`) and the other occurrences of its tag still get their values. `False` restores the 0.7 order.
- `synthetic_low_confidence_locale` (default `'en_US'`, legacy module global `SYNTHETIC_LOW_CONFIDENCE_LOCALE`): with `synthetic_locale=None`, a document whose language detection gets the `language_low_confidence` warning uses this Faker locale instead of the detected language's. Before, `Kontakt: Anna Schmidt, anna.schmidt@firma.de, Telefon 030 1234567.` was detected as Danish (0.17) and got Danish fakes. Deterministic across processes as before. `None` follows the detected language anyway.
- `synthetic_collapse_period` (default `True`, legacy module global `SYNTHETIC_COLLAPSE_PERIOD`): a synthetic fake ending in `.` drops that period when the text already has a `.` right after the entity (de_DE names with a suffix such as `B.Eng.` gave `B.Eng..` before a full stop). `False` keeps both periods.
- `sct-clean --preset {sct,training}` (default `sct`) and `SCT_MCP_PRESET` / `create_mcp_server(preset=...)` (default `sct`) pick the config preset; `training` keeps numbers, case and whitespace as written. `sct-clean --to-ascii` / `SCT_MCP_TO_ASCII=1` turn on ASCII folding.
- `onnx_intra_op_threads` and `onnx_inter_op_threads` (default `None`, the ONNX Runtime default, same as 0.7.1). Legacy module globals `ONNX_INTRA_OP_THREADS` / `ONNX_INTER_OP_THREADS`. Not applied to GLiNER's own ONNX session (`gliner_onnx=True`).
- `SCT_MCP_ROOT` (default unset = off) and `SCT_MCP_MAX_BYTES` (default 1 MiB): let MCP `clean_text(path=...)` read regular UTF-8 files under the root. Relative paths only; `..`, hidden parts and symlinks are refused (opened folder by folder with `O_NOFOLLOW`); off on Windows.
- `ScoreConfig` (`sct.score`, not `TextCleanerConfig`): `quality=False`, `quality_model=None`, `quality_revision=None`, `minhash=False`, `minhash_num_perm=128`, `minhash_seed=1`, `minhash_shingle=5`, `minhash_min_tokens=5`, `data_card=False`.
- Adapter options `columns`, `output` (`"overwrite"`), `config` (`None` = `"training"`); a single string for `columns` is one column.
- `ner_chunk_overlap` (default `0`, legacy module global `NER_CHUNK_OVERLAP`): each NER window adds this many tokens of the neighbouring text on both sides of its chunk (GLiNER-only: 2 characters per token), and a span is kept from the window whose chunk holds its start; a neighbouring window's span is kept too when no span of the owning window that passes the confidence threshold (and, with a privacy policy, is replaced or redacted by it) covers it. A span seen by several windows is one vote per model, with its best score and that score's label, so repeated context (even with a different label) never counts as extra models in ensemble voting. A document that fits one window is never split, so overlap only affects documents that are chunked. This finds entities that a chunk boundary would split (sentence, whitespace or hard cut). The default `0` keeps the 0.7.1 chunks; overlap changes the context the model sees in long documents, so detections can change either way, and inference grows by about `2 * overlap / window`. An entity longer than the overlap that starts at the very end of a chunk can still be cut. Must satisfy `2 * overlap < window // 2`; this is checked again for each document, so a model loaded later with a smaller limit raises `ValueError` instead of running with a negative chunk budget.

### Fixes

- NER no longer tags text inside a regex placeholder: with the default models, `call +1-202-555-0147` became `call <<MISC>>` (the model tagged `PHONE` in `<PHONE>`; 0.7.1 did the same). Spans inside an upper-case `<TAG>` / `<TAG_n>` marker are dropped; custom `replace_with_*` text is not protected.
- International phone numbers no longer swallow a closing parenthesis of the sentence: `(tel +31 6 1234 5678)` now gives `(tel <PHONE>)`, not `(tel <PHONE>`. Area-code parentheses must be balanced (`+44 (0)20 ...` unchanged).
- `ner_backend='presidio_gliner'`: a detected language that the Presidio analyzer has no NLP engine for is analyzed as English instead of failing (the default `AnalyzerEngine` only has English).
- A warning is logged when CUDA was requested but ONNX Runtime fell back to CPU (missing CUDA/cuDNN libraries or a driver that does not match the onnxruntime-gpu build). Before, this happened silently. New `docs/PERFORMANCE.md` gives CPU and GPU throughput and CUDA setup notes.
- NER windows now fit every tokenizer of the models a document's language runs, not only the one with the smallest limit. Models loaded for other languages do not change a document's windows. With the default models, the multilingual model (WordPiece) can need far more tokens than XLM-RoBERTa for the same text and silently truncated the rest of a window, so it never saw entities past that point. Such texts now get more, smaller windows; on the release set 2 of 30 documents (and 2 of 4 joined long documents) are affected. Default-config output of the release set is unchanged, but these documents run more model calls (single-run Mac benchmark of the release set: 6.9 to 5.4 docs/s).
- Overlapping detections (for example `John Paul Smith` and `Paul` from two ensemble models) are merged into one span before replacement in reversible and synthetic mode and when a custom label is present. Before, right-to-left replacement left part of the outer entity visible (`<PERSON_1> Smith`). Placeholder mode with standard labels already merged them through Presidio. In the release benchmark, reversible output of `zh-glued-001` changes from the garbled `<LOCATION_1>OCATION_0>` to `<LOCATION_0>`.
- ONNX models exported with a fixed batch dimension of 1 run one row per call instead of failing under `ner_batch_size > 1`.
- `replacement_mode='synthetic'`: the regex steps and NER now share one consistency cache per document under `ner_batch_size > 1`, so the same value replaced by both gets the same fake value (as with `ner_batch_size=1`).
- The INT8 quantized cache directory name is collision-free: commits are hashed and repo ids that contain `__` are %-quoted.
- ONNX models with external weights (the default XLM-RoBERTa-large models) load from a huggingface-hub cache that stores each blob in its own directory (`hub/blobs/<xx>/<sha256>`, seen with huggingface-hub 1.32.0). onnxruntime refused them with "External data path escapes model directory"; 0.7.1 fails the same way. In that layout SCT now hard-links `model.onnx` and `model.onnx_data` into `~/.cache/sct_linked/<model>@<commit hash>` (under `cache_dir` when set; one directory per snapshot, so a branch that moves back to an already-cached commit loads that commit's weights) and loads them from there; if hard links are not possible (another filesystem) the files are copied once (about 2.2 GB per model) and a warning is logged; later loads, and other processes starting at the same time, reuse the copy while its size matches and its source blob is not newer. Other cache layouts load in place as before.
- Chunk planning ignores fixed padding set in a model's `tokenizer.json`. Such a tokenizer counted every text as its padded length, so NER windows were split down to single characters and entities were missed (0.7.1 had this for the primary tokenizer; 0.8.0 had spread it to every ensemble tokenizer). None of the default models set padding.
- Custom `replace_with_numbers` / `replace_with_currency_symbols` markers are inserted literally; before, a backslash in them was read as a regex group reference.
- Linux worker processes: freeing a cleaner (its ONNX session) inside a forked child crashed (exit 139) or hung in garbage collection, for example in Hugging Face `num_proc` or DataTrove workers. Live NER objects are now pinned in the child at fork (`os.register_at_fork`) and never destroyed there. `forkserver` was not affected.
- Framework adapters build one cleaner per handle also when several threads ask at once; the lock is replaced after a fork so a child never deadlocks.
- Hugging Face `clean_dataset` returns only the columns SCT wrote, so `remove_columns=["text"]` with `output="new_column"` really drops the raw text, and a stale input column named like an output column is replaced. Its cache fingerprint is stable across runs (`datasets` hashed a frozenset in per-process order).
- NER windows respect a tokenizer's saved truncation `max_length` when it is below `model_max_length`, so inference never cuts a planned window.
- A language route given as a list (`["ENGLISH", ...]`) is no longer changed in place when GLiNER is added to it.
- `scripts/benchmark_release.py` imports on Windows (no `resource` module); peak RSS is reported as `null` there.
- Isolated-symbol removal (library default, off in `training`) only removes symbols with whitespace or a text edge on both sides. Before, a `)` or full stop after a placeholder or bracket was dropped: `(born 1980).` became `(born <YEAR>`, `bel +31 20 123 4567.` lost its full stop, and a Chinese `，` after a word was dropped (0.7.1 did the same). On the release benchmark 11 of 30 outputs now keep such a `.` or comma; scores are unchanged.
- `replacement_mode='synthetic'` with NER: the model no longer rewrites a fake email or phone that a regex step just inserted (`anna@beispiel.de` became `James Howard@Bernard LLC`); those fakes are protected like placeholders.
- WordPiece NER models (any tokenizer with a `##` continuation prefix, such as `protectai/bert-base-NER-onnx`): entities widen to whole words, so `John Doe` no longer becomes `<PERSON>e`. SentencePiece models (the defaults) are unchanged.
- Worker processes no longer call the Hub when the cache is warm. After `model.onnx` resolves, the other files load from that commit (served from the cache with no request) and the missing `model.onnx_data` probe uses the Hub's cached "no such file" answer (the default ensemble made one HEAD per load, an unpinned model eight). A `CleanerHandle` that is forked or pickled after loading pins unpinned models to the commits the parent loaded. A forked worker on Linux could hang forever in DNS (`getaddrinfo`) during that request.
- `language_confidence` is rounded to 6 decimals. Lingua's value differed in the last digit between processes, so worker and in-process metadata did not match.
- `import sct` no longer imports `presidio_anonymizer` (which pulls `presidio_analyzer`, GLiNER and torch when installed); it loads on first NER replacement.
- MCP: an `SCT_MCP_ROOT` / `create_mcp_server(root=...)` that is not an existing directory raises `ValueError` at start-up instead of failing on every call.
- `WindowedStreamingRedactor` checks for a split contact token on HTML-entity-decoded text too, so a phone written with `&#32;`, `&#x20;` or `&nbsp;` between its digit groups is no longer cut between windows.
- `ner_chunk_overlap`: window offsets come from the tokenizer of the document's language route, not always the primary model's tokenizer.
- A worker forked while another thread was building the language detector gets a fresh detector lock instead of inheriting it held (a possible hang).
- URLs whose host is a private or loopback IPv4 address (`10.x`, `127.x`, `172.16-31.x`, `169.254.x`, `192.168.x`) or a bracketed IPv6 address (`[::1]:8080`) become one `<URL>`. Before, such a URL was not matched as a URL and query values such as `?email=alice%40example.com` stayed in the text.
- URLs keep interior `)`, `]` and `}` when more URL text follows (`?f=(a)&email=...`), so the rest of the query is replaced too; a bracket closing around a URL still stays outside it.
- NER: when one model gives two labels for the same span (repeated window context, WordPiece widening), it is one vote with its best score and that score's label, also without `ner_chunk_overlap`. Before, the two labels could average a 0.99 `PER` below the confidence threshold.
- `sct.integrations.hf.clean_dataset` declares the output column features, so a first batch of only `None` or `[]` no longer types a column as null and fails on the next batch.
- `assess_reidentification_risk` type-tags values inside tuples and frozensets too, so `(1,)` and `(True,)` are different quasi-identifier values.
- Merged signals of a JSON value: `language_source` comes from the longest leaf of the chosen language, not the longest leaf overall.
- The email domain stops at general, CJK and full-width punctuation. With `fix_character_width=False` (the new default) `seikyu@example.com（担当：鈴木花子）` was one `<EMAIL>` up to `）`, so the name inside was hidden from NER and labelled EMAIL; now the address ends at `.com`. Domains in any script (`例え.jp`, `उदाहरण.भारत`) still match, and the letters of the CJK punctuation block (`々 〆 〇`, by Unicode category) stay domain letters (`info@佐々木.jp`), as do the join controls ZWNJ and ZWJ (U+200C/D) that Persian and Devanagari domains use.
- Isolated-symbol removal keeps a quote, dash, `*` or `%` that touches a letter, digit or mark of any script. Before only ASCII letters and digits counted, so `東京-大阪` became `東京大阪` and `ইসলাম-এর` became `ইসলামএর` (which also made a WordPiece model widen a name over the suffix).
- Email local parts take combining marks (`প্রিয়া@example.com`, `संपर्क@example.com` were left whole or cut to `संपर्<EMAIL>` without ASCII folding); full-width `＠ ．` after a mark fold too. Email and URL hosts take letters beyond the BMP (`info@𠮷野.jp`) and punycode TLDs (`example.xn--p1ai` was cut at `.xn`); a URL with an IPv4 host takes every valid octet (`http://10.0.0.0/?...`, `.255`) and only a whole address, so query values no longer stay visible.
- Phones and IPs are matched on a view where every decimal digit is ASCII and full-width `. -` are ASCII, with the source offsets kept: `٠٦ ١٢ ٣٤ ٥٦ ٧٨`, `٩٨٧٦٥ ٤٣٢١٠`, `١٩٢．١٦٨．١．١` are replaced and `1.2.3.4١` is one `<IP>`.
- `dataclasses.replace(cfg, legacy_type_names=...)` on a `ner_mode="pii"` config re-derives the label map and positional tags (an explicit `gliner_label_map` stays); the `presidio_gliner` backend maps analyzer types through `gliner_label_map` and `PHONE_NUMBER`/`IP_ADDRESS` to `PHONE`/`IP` unless `legacy_type_names=True`.
- With `ner_sees_tags`, a regex value stays hidden through the cleanup steps, so a tag they remove (`replace_with_email="[EMAIL]"` under bracket removal) no longer loses the value in reversible mode. A forked child gets a fresh synthetic random lock. `llms.txt` lists the current `check_replace_dates`, `language_confidence_warning` and `language_margin_warning` defaults.
- `AGENTS.md` (contributors): no word-list fixes for model mistakes; every behaviour is a safe default plus a config option.

### Known limits

- In ensembles with GLiNER, windows are sized for the token-classification models, not for GLiNER's own word limit.
- Only `process_batch` is faster. `process` of one document is not batched across documents.
- GLiNER2 runs one chunk per call; torch models and GLiNER keep one lock per model, so they do not run in parallel across threads.
- `ner_chunk_overlap` (off by default) can still cut an entity longer than the overlap that starts at the very end of a chunk.
- The email regex takes CJK text glued before an address (`请发送至wangfang@example.com` is one `<EMAIL>`) and an Arabic comma `،` after it. No PII leaks, but spans and nearby text are wrong. `06-12345678` becomes `06-<PHONE>`.
- ORGANISATION is the weakest NER label in the multilingual eval (0.69 recall), mostly in non-Latin scripts; Bengali and Tamil name spans can be a few characters off. WordPiece widening makes the mBERT and XLM-RoBERTa spans agree more often, and the ensemble averages their scores, so a weak mBERT score can drop an agreed span below the threshold: some ar/bn/ta organisation names and 1 Thai name masked by the pre-review 0.8.0 code are not masked now (`docs/MULTILINGUAL_EVAL.md`).
- A URL with a path swallows a trailing sentence period (`see https://x.org/a.` becomes `see <URL>`).
- Adversarial strings such as `+000000000(000)0` can leave `(000)0` after phone replacement (Python `re` takes the first alternative that matches, not the longest).
- Score B's quality head was tested with a small ONNX graph, not the exported FineWeb2-HQ-Plus model; its speed is not measured. GLiNER PII mode (`ner_mode="pii"`) was not run end to end for this release.
- An unpinned NER model makes one Hub request per worker when the parent never loaded it before forking; pin `ner_model_revisions` or set `HF_HUB_OFFLINE=1`.
- With an empty regex marker (`replace_with_email=""`) and `ner_sees_tags`, NER in synthetic/reversible mode reads the indexed placeholder or fake, not the empty text placeholder mode leaves, so its findings can differ there.
- CI does not install the `mcp` extra; FastMCP-only tests skip there.

## 0.7.1 - 2026-10-30

This release fixes PII that 0.7.0 left in the output. On the release benchmark
(`evaluation/results/comparison-0.7.0-0.7.1.md`) exact-span recall goes from 0.62 to 0.98
and leaked PII values from 6 of 67 to 0.

### Behaviour changes

- More PII is found and replaced: glued and international phone numbers, IP addresses, and GLiNER findings between 0.3 and 0.85 in PII mode. Output text changes for inputs that contain them.
- IP addresses are now replaced with `<IP>` by default (`<IP_ADDRESS_n>` in reversible mode), including private ones.
- New config flags: `check_replace_ips` (default `True`), `replace_ips_public_only` (default `False`) and `replace_with_ips` (default `"<IP>"`). They are the last `TextCleanerConfig` fields, so positional arguments keep their 0.7.0 meaning.
- The browser manifest rejects replacement tokens that contain digits, or that a later enabled rule would rewrite (for example an HTML token holding an email); the browser has no protection for emitted tokens.
- `process_json` finding paths for dict keys use `["@key", <index>]` instead of the key text.

### Fixes

- Phone and year rules now find digits glued to letters in any script (for example `电话9876543210`, `वर्ष2023में`).
- The number rule now finds numbers glued to letters in scripts written without spaces (`金额5000元`). Digits inside Latin, Greek and Cyrillic words stay as they are (`GPT-4o`, `Qwen3-8B`, `H2O`). Set `replace_numbers_in_words=True` to replace those too (`GPT4` becomes `GPT<NUMBER>`).
- IPv4 and IPv6 addresses (public and private) are replaced with `<IP>` before the date, phone and year steps. Set `replace_ips_public_only=True` to keep private, loopback and reserved addresses, or `check_replace_ips=False` to turn the step off.
- Phones now run before years, so `+31 6 1234 2019` and `2001:4860:4860::8888` are no longer split by the year rule. The browser manifest uses the same order.
- IPv6 is matched in every valid compressed form, including IPv4 tails (`::abcd:192.0.2.1`, `1:2:3:4:5::192.0.2.1`), and invalid forms are not matched, also in the browser rule.
- Replacement tokens and synthetic values (`<EMAIL1>`, a synthetic IP or phone) and IPs kept by `replace_ips_public_only` are no longer rewritten by the later phone, year and number steps.
- Phone detection covers more international formats (`+34 612 345 678`, `+91 98765 43210`) and Indian mobiles (`98765 43210`, `098765-43210`).
- Oversized NER chunks are split again, also long Chinese, Japanese and Hindi text without spaces: to the tokenizer's token budget for ONNX/torch models, and to a cap of `2 * max_tokens` characters in GLiNER-only mode (a character estimate, not a token count; token-budget chunking is planned for 0.8.0). Text is split after `。！？` and `। ॥`, and reversible map offsets for NER entities are now document-relative. Long whitespace runs at a chunk cut no longer break chunk alignment. Chunk planning ignores truncation set in a model's `tokenizer.json`, and a slice that grows when re-tokenized is split again, so text past the model limit is no longer left out of model input.
- Languages the `stop-words` package does not support (for example `THAI`) no longer crash `TextCleaner` creation.
- PII mode: the final `ner_confidence_threshold` now follows `gliner_threshold` (0.3 by default) instead of a hidden 0.85, so GLiNER findings between 0.3 and 0.85 are no longer dropped. An explicit `ner_confidence_threshold` other than 0.85 still wins (0.85 equals the default and cannot be told apart), and `0.0` is no longer turned into 0.85.
- `process_json` and MCP `clean_json` no longer return original dict keys in finding paths: a key's own findings use `["@key", <index>]`.
- `is_sensitive_result` compares placeholder counts, so `<EMAIL> alice@example.com` (HTML step off) is reported as sensitive. Marker-like tags removed by the HTML step (`<EMAIL> Alice` with `replace_with_html=""`) no longer cancel a new placeholder.
- `TextCleanerConfig` can be pickled and deep-copied, and the copy keeps the exact state. `dataclasses.replace(cfg)` no longer grows the default 5 NER models to 8 and 4 supported languages to 7. A partial `ner_models` mapping gets the same defaults whatever its mapping type.

## 0.7.0 - 2026-09-19

SqueakyCleanText now supports privacy workflows beyond single-text cleanup.
Applications can apply policies, inspect decisions, and use consistent adapters.

### New features

- Added versioned privacy policies, structured findings, detector provenance, and metrics callbacks.
- Added external token storage for reversible anonymization maps.
- Added JSON, DataFrame, RAG, command-line, and MCP adapters.
- Added exact-span PII evaluation with language, domain, label, and custom slice reports.
- Added pinned model candidates, benchmark tooling, and strict model adoption gates.
- Added buffered streaming, OCR image redaction, browser manifests, and corpus risk indicators.

### Improvements

- Pinned default ONNX, PyTorch, GLiNER, and GLiClass model revisions.
- Preserved source offsets across normalized NER chunks.
- Made `process_batch(batch_size=...)` control document concurrency.
- Added Python 3.14 CI and an 85 percent coverage gate.
- Added optional backend CI, standard lockfiles, and PyPI Trusted Publishing.
- Updated reversible mode to restore emails, phone numbers, URLs, dates, numbers, HTML, and currencies.

### Fixes

- HTML replacement now honors its configured replacement token.
- Presidio now receives the detected document language.
- Default configuration no longer emits a deprecation warning.
- MCP findings no longer return original entity values.
- Overlapping policy findings now redact their complete combined span.

### Migration notes

- `gliclass_onnx=True` now raises an error because upstream GLiClass has no native ONNX loader.
- Structured adapters reject reversible mode when they cannot return each token map safely.
- Reversible fuzzy date replacement remains unsupported and now fails during configuration.

## 0.6.1

- Improved GLiNER ONNX fallback behavior and test infrastructure.

## 0.6.0

- Added PII mode, synthetic replacement, reversible anonymization, and document classification.
