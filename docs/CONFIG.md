# Configuration

Every step of the pipeline is a `TextCleanerConfig` field. The pipeline is built once when the
`TextCleaner` is created; a step that is switched off is skipped entirely (zero overhead).
Each step has a safe default and an option to change it.

> **Start with a preset.** `preset_config("training")` for training or RAG data, or the default
> config; see [Presets](#presets). Use the [field reference](#field-reference) only to change one field.

## Using TextCleanerConfig

```python
from sct import TextCleaner, TextCleanerConfig

# Create an immutable configuration
cfg = TextCleanerConfig(
    check_ner_process=True,
    ner_confidence_threshold=0.85,
    positional_tags=('PER', 'LOC', 'ORG', 'MISC'),
    replace_with_url="<URL>",
    replace_with_email="<EMAIL>",
    replace_with_phone_numbers="<PHONE>",
    language="en",  # Pin to English (also accepts 'ENGLISH', 'eng')
)

# Initialize with config
cleaner = TextCleaner(cfg=cfg)
```

## Presets

`preset_config("training")` (also `TextCleanerConfig.from_preset()`) only removes PII and
repairs broken unicode, for LLM training and RAG corpora: URLs, emails, phones, IPs and NER
entities become placeholders; numbers, years, dates, HTML, brackets, case, punctuation,
line breaks, indentation and backslash escapes in code are kept, and no `stat_text` is made.
`"sct"` is the library default config. Any config field overrides the preset:

```python
from sct import TextCleaner, preset_config

cleaner = TextCleaner(cfg=preset_config("training", check_replace_numbers=True))
```

The data-pipeline adapters use `"training"` when no config is given. GLiNER PII stays opt-in
(`ner_mode="pii"`).

## Languages

All language parameters accept Lingua names (`'ENGLISH'`), ISO 639-1 (`'en'`), or ISO 639-3 (`'eng'`) codes:

```python
from sct import TextCleanerConfig

# Pin to one language (skip auto-detection)
cfg = TextCleanerConfig(language='de', check_ner_process=False)

# Restrict detection to specific languages (auto-detect among them)
cfg = TextCleanerConfig(language=('en', 'nl', 'de'), check_ner_process=False)

# Add extra languages (stopwords/dates/NER keys)
cfg = TextCleanerConfig(extra_languages=('fr', 'pt'), check_ner_process=False)

# Pin any of the 75 Lingua languages, e.g. Hindi
cfg = TextCleanerConfig(language='hi', check_ner_process=False)

# Bring your own NER model for a language: Hindi text runs only this model
cfg = TextCleanerConfig(ner_models={'ENGLISH': '...', 'MULTILINGUAL': '...', 'HINDI': 'your/hindi-ner-onnx'})
```

Auto-detection covers all 75 Lingua languages (`detect_all_languages=True`). Languages
without their own NER model run the MULTILINGUAL + ENGLISH fallback; stopwords exist for
EN/NL/DE/ES (others use English) and month names for 8 languages (EN NL DE ES FR PT IT TH) plus `custom_month_names`.
`ner_backend='gliner'` is language independent, so any pinned language works with it.

## Field reference

The same fields with their docstrings, generated from the code: [TextCleanerConfig API page](https://rhnfzl.github.io/SqueakyCleanText/reference/config/).

Immutable (frozen) dataclass. Create modified copies with `dataclasses.replace()`:

```python
import dataclasses
from sct import TextCleanerConfig

cfg = TextCleanerConfig()
new_cfg = dataclasses.replace(cfg, check_ner_process=False)
```


**Pipeline toggles** (all `bool`, default shown):

| Field | Default | Description |
|-------|---------|-------------|
| `check_detect_language` | `True` | Auto-detect language |
| `check_fix_bad_unicode` | `True` | Fix encoding issues via ftfy and decode HTML entities (`&amp;`, `&#64;`), each on its own, so `<p>john&#64;example.com</p>` is replaced. Escaped `&lt;` / `&gt;` decode only outside tags, after tag removal, so `&lt;b&gt;` stays the text `<b>`. `False` decodes nothing |
| `check_decode_unicode_escapes` | `True` | Part of `check_fix_bad_unicode`: decode literal backslash escapes (`\u00e9` to `é`, `\n` to a line break). `False` keeps them as written (code, JSON) |
| `check_to_ascii_unicode` | `False` | Transliterate to ASCII with unidecode. `False` keeps every script (`日本語`, `München`); the multilingual NER models read native script, and folded output hid names from them (0 of 8 Hindi/Bengali/Tamil names found). `True` = 0.7 (`日本語` -> `Ri Ben Yu`); `sct-clean --to-ascii`, `SCT_MCP_TO_ASCII=1` |
| `check_replace_html` | `True` | Strip/replace HTML tags, found on the markup as written: escaped code (`&lt;b&gt;`) is text, not a tag |
| `html_tag_strict` | `True` | The HTML rule reads only tag shapes: a name after `<` (`<b>`, `</a>`, `<br/>`, `<!DOCTYPE html>`, `<?xml ...?>`, multi-line tags) or a `<!-- -->` comment, plus entities. `Name <a@b.com>`, `<https://...>` and a stray `a < b ... >` are left to the email/URL rules and the text. `False` restores the 0.8.0 `<.*?>`, which masks those as `<HTML>`. Legacy global `HTML_TAG_STRICT` |
| `url_trailing_punct_outside` | `True` | `. , ; : ! ? '` end a URL path or query unless more URL text follows, so `see https://x.com/a/b. Next` gives `see <URL>. Next` and the sentence keeps its full stop or comma. `/a.html`, `?a=1,2&b=3`, `/v1.2/x`, `;jsessionid=` and a fully qualified host's final dot before a path stay inside. Format regex only; reaches the plain, reversible and synthetic paths, the streaming split guard and the browser export. `False` restores the 0.8.0 URL, which takes the trailing mark. Legacy global `URL_TRAILING_PUNCT_OUTSIDE` |
| `check_replace_urls` | `True` | Replace URLs with token |
| `check_replace_emails` | `True` | Replace emails with token. Text with no `@` and no bracketed `at` (`[at]`, `(at)`, `<at>`, `{at}`) skips the email regex; output is unchanged |
| `check_replace_ips` | `True` | Replace IPv4/IPv6 addresses (runs before dates, phones, years) |
| `replace_ips_public_only` | `False` | Keep private, loopback and reserved IPs; replace only public ones |
| `check_replace_years` | `True` | Replace years (1900-2099) |
| `check_replace_dates` | `True` | Replace full dates and date-times with one `<DATE>` (ISO 8601, numeric with `-` `/` `.` in digits of any script and any 4-digit year, so Buddhist-era `15/01/2533` too, month names (Thai with Buddhist-era years when passed via `custom_month_names`, `24 ก.พ. 2562`), month name / year (`October/55`), CJK/Korean unit dates `1990年1月15日` `1990년 1월 15일`, Korean `1990. 1. 15.`, Han-numeral `一九九〇年一月十五日` read by Unicode numeric value); runs before phones, years and numbers. A date ends where no digit or Latin/Greek/Cyrillic letter touches it, so one glued to Thai or CJK text is found (`生日是15 January 1990谢谢`). Only full dates become one `<DATE>`: `1990年1月`, `1月15日`, Japanese era years (`令和6年1月15日`), `1990.1.15.` (CalVer-shaped) and month names outside the 7 built-in languages (add them with `custom_month_names`) do not, and the year and number steps can still change them (`1990年1月` becomes `<YEAR>年1月`, or `<YEAR>年<NUMBER>月` with `replace_numbers_mode='all'`). `False`: digits fall to the year and number steps |
| `date_month_name_slash_year` | `True` | A month name, `/` and a 2- or 4-digit year is one `<DATE>` (`October/55`, `Oct/2019`). `False` = 0.8.0 (`October/<NUMBER>`) |
| `date_builtin_month_languages` | `None` | ISO codes of the built-in month data dates use, from `en nl de es fr pt it`; `None` = all (the 0.8.0 data) |
| `check_fuzzy_replace_dates` | `False` | Fuzzy match misspelled months (requires `[fuzzy]`) |
| `check_replace_phone_numbers` | `True` | Replace phone numbers |
| `check_replace_numbers` | `True` | Replace numbers (which ones: `replace_numbers_mode`) |
| `replace_numbers_mode` | `'identifiers'` | `'identifiers'`: only whole digit runs of an identifier shape become `<NUMBER>` (`Order 4711`, `#12345`, `ID:98765`, digits of any script). Kept: amounts and decimals (`1,250.00`, `12.5`), percents and numbers beside a currency sign (`12345%`, `€12345`), times and ranges (`14:30`, `1000-3000`), runs shorter than `replace_numbers_min_digits`, and digits that are part of a longer token: glued to a Latin/Greek/Cyrillic letter, digit or `_`, or joined by `- / + * . _ \| ,` or a dash to one (`ID55555`, `x-55555`). A code like `INV-2023-107683` gets no partial `<NUMBER>`: the number step leaves it whole (the year step still takes `2023` when `check_replace_years` is on). Reversible mode also masks a run glued to a placeholder that stands alone (see REPLACEMENT_MODES). Letters of unspaced scripts do not glue (`金额50000元` gives `金额<NUMBER>元`). A whole digit-only token of three or more hyphen-joined groups with `replace_numbers_min_digits` digits in all is one `<NUMBER>` (`123-45-6789`); date-shaped ones (`2023-11-05`, `05-11-23`) are kept. On the benchmark DEV split the old rule caused 62-68% of false masks. `'all'` = 0.8.1, every number. Reaches the plain, reversible and synthetic paths, `process_batch`, the streaming redactors and the browser export. Legacy global `REPLACE_NUMBERS_MODE` |
| `replace_numbers_min_digits` | `4` | Fewest digits an identifier-shaped number needs (`'identifiers'` mode). 4 gave the best F2 of 4 to 7 on DEV (0.881, recall 0.893, precision 0.856 with years and dates off and `phone_reject_embedded`). Legacy global `REPLACE_NUMBERS_MIN_DIGITS` |
| `replace_numbers_in_words` | `False` | Letters do not end a number: in `'all'` mode digits inside Latin/Greek/Cyrillic words are replaced (`GPT4` to `GPT<NUMBER>`); in `'identifiers'` mode a letter beside a digit run no longer makes it part of a longer token (`ID55555` to `ID<NUMBER>`) |
| `phone_reject_embedded` | `True` | A digit-only phone match glued to a Latin/Greek/Cyrillic letter or `_`, directly or across a joiner (`ID9876543210`, `ref-0612345678`), or that is exactly a yyyymmdd date or yyyymmddhhmm[ss] timestamp (`20231105123456`) is not a phone (`2023010123` still is). Formatted and international phones (`06 1234 5678`, `+31 6 1234 5678`, `06-12345678`), `Tel:0612345678` and phones glued to unspaced scripts (`电话9876543210`) are unchanged. DEV: precision +0.02, recall -0.01. Server-side only: the browser phone rule keeps 0.8.1 matches. `False` = 0.8.1. Legacy global `PHONE_REJECT_EMBEDDED` |
| `numbers_letter_prefix_ids` | `True` | `'identifiers'` mode: a whole token of 1 to 3 uppercase Latin letters, 6+ digits of any script and 0 to 2 uppercase letters is one `<NUMBER>` (`AB1234567`, `X12345678Z`), not a currency code before digits (`USD1000000`). Lowercase (`ab1234567`) and 4+ letters are kept. Not tied to `replace_numbers_min_digits`. `False`: the digits stay (glued to letters). Legacy global `NUMBERS_LETTER_PREFIX_IDS` |
| `numbers_two_part_ids` | `True` | `'identifiers'` mode: two digit groups joined by `/` or a hyphen, the first 4+ digits and 7+ in all, with an optional uppercase check letter (`123456/7890`, `12345-678901`, `123456-789K`, `Az. 12345/67`), or by `.` with 4+ digits on each side (`12345.67890`), are one `<NUMBER>`. Kept: year pairs (`2023-2024`), ranges whose groups both end in `00` (`1000-3000`), a time after it (`1234-5678:30`), a `%` or currency sign beside it. A range like `12345-67890` cannot be told from an ID and is masked. Legacy global `NUMBERS_TWO_PART_IDS` |
| `numbers_spaced_pair_ids` | `True` | `'identifiers'` mode: `DDD DD` as two whole tokens (a 5-digit postal code written `123 45`) is one `<NUMBER>`, and so is a whole run of 3 or more digit groups joined by spaces and/or hyphens (at least one space) with 7+ digits (`123 45 6789`, `123 45-6789`, `1234 56-78-90`, `12 34 56 78 90 12 34`), never only part of it. Thousands-grouped amounts of up to 6 groups (`1 234 567`) and runs beside a decimal, `%` or currency sign stay. The windowed streaming redactor may split a run of 14+ groups, which then comes out as several `<NUMBER>` tokens. A date written with spaces (`05 11 2023`) is not a `<DATE>` (the date step has no spaced shape): with years on its year is `<YEAR>`, with years off the run is one `<NUMBER>`. Legacy global `NUMBERS_SPACED_PAIR_IDS` |
| `check_replace_currency_symbols` | `True` | Replace currency symbols |
| `check_ner_process` | `True` | Run NER entity recognition |
| `check_remove_isolated_letters` | `True` | Remove single letters |
| `check_remove_isolated_special_symbols` | `True` | Remove isolated symbols (reversible mode keeps a quote, dash, `*` or `%` touching a placeholder, so it restores) |
| `check_remove_bracket_content` | `True` | Remove `[...]` content (needs `check_remove_isolated_special_symbols`) |
| `check_remove_brace_content` | `True` | Remove `{...}` content (needs `check_remove_isolated_special_symbols`) |
| `bracket_content_max_chars` | `32` | Bracket/brace removal only takes innermost `[...]`/`{...}` spans with at most this many characters inside, so bracketed prose stays. `0` = no limit |
| `bracket_content_max_words` | `3` | Same, at most this many whitespace-separated words inside. Both limits `0` = 0.8.0 (every span up to the first closing bracket) |
| `keep_json_braces` | `True` | Never remove a `{...}` that parses as a JSON object (nested ones included), nor anything inside it. `False` = 0.8.0 |
| `check_normalize_whitespace` | `True` | Normalize whitespace |
| `check_statistical_model_processing` | `True` | Generate stat model output |
| `check_casefold` | `True` | Lowercase stat output |
| `check_smart_casefold` | `False` | Lowercase but preserve NER tokens |
| `check_remove_stopwords` | `True` | Remove stopwords from stat output |
| `check_remove_punctuation` | `True` | Remove punctuation from stat output |
| `check_remove_stext_custom_stop_words` | `True` | Use your `custom_stopwords` sets in `stat_text`. `False` keeps the built-in stopword lists |
| `check_remove_emoji` | `False` | Remove emoji characters |

**Replacement tokens** (all `str`):

| Field | Default |
|-------|---------|
| `replace_with_url` | `"<URL>"` |
| `replace_with_html` | `"<HTML>"` |
| `replace_with_email` | `"<EMAIL>"` |
| `replace_with_ips` | `"<IP>"` |
| `replace_with_years` | `"<YEAR>"` |
| `replace_with_dates` | `"<DATE>"` |
| `replace_with_phone_numbers` | `"<PHONE>"` |
| `replace_with_numbers` | `"<NUMBER>"` |
| `replace_with_currency_symbols` | `None` |

**Type names**: each PII type has one name on every surface: the tag (`<PHONE>`), the reversible placeholder (`<PHONE_0>`), `metadata['regex_counts']` / `entity_counts`, `signals['pii_types']`, score columns, DataTrove `pii_<TYPE>` stats, MCP output and PII-mode NER tags. Regex types: `URL`, `HTML`, `EMAIL`, `IP`, `DATE`, `YEAR`, `PHONE`, `NUMBER`, `CURRENCY`; NER types: `PERSON`, `LOCATION`, `ORGANISATION`, `MISC` (PII mode adds the upper-cased GLiNER label, e.g. `CREDIT_CARD_NUMBER`). `legacy_type_names=True` restores the pre-0.8.0 names `PHONE_NUMBER` / `IP_ADDRESS` everywhere except the tag text (set that with `replace_with_*`).

**NER settings**:

| Field | Default | Description |
|-------|---------|-------------|
| `ner_backend` | `'onnx'` | Backend: `onnx`, `torch`, `gliner`, `ensemble_onnx`, `ensemble_torch`, `presidio_gliner` |
| `ner_mode` | `'standard'` | `'standard'` or `'pii'` (auto-configures GLiNER for PII detection) |
| `gliner_onnx` | `False` | Load the GLiNER model from its pre-built ONNX weights on the Hub (uni-encoder models only). PII mode sets it when `ner_backend` is `'onnx'` |
| `gliclass_onnx` | `False` | Must stay `False`: the gliclass package has no ONNX loader, so `True` raises `ValueError` |
| `replacement_mode` | `'placeholder'` | `'placeholder'`, `'synthetic'` (Faker), or `'reversible'` (indexed placeholders + deanonymize map) |
| `synthetic_seed` | `42` | Base seed for `replacement_mode='synthetic'` |
| `synthetic_per_document` | `True` | Seed each document's fakes from `synthetic_seed` + its text: fakes differ across documents, the same document always gets the same fakes (any worker count or order). `False` = 0.7: one Faker stream per thread, so batched documents could all get the same first fake |
| `synthetic_locale` | `None` | Faker locale for synthetic fakes. `None` = follow the detected document language (Dutch text gets Dutch names and cities, Chinese text Chinese names; `en_US` when no language is detected or Faker has no locale for it, e.g. Norwegian Bokmal, Malay, Urdu). A string forces one locale; `'en_US'` = 0.7 |
| `ner_sees_tags` | `True` | In `synthetic` and `reversible` mode NER reads the same text as placeholder mode (`<EMAIL>`, `<PHONE>`, `<URL>`...); the fakes or indexed placeholders (`<EMAIL_0>`) go in after NER, so every mode finds the same entities; custom markers (`replace_with_email="EMAIL"`) are kept out of NER spans like `<EMAIL>`. `False` = 0.7 order: NER reads the fakes / indexed placeholders, which can change what it finds |
| `synthetic_low_confidence_locale` | `'en_US'` | With `synthetic_locale=None`: Faker locale for documents whose language detection gets the `language_low_confidence` warning, so a short German line misdetected as Danish does not get Danish fakes. `None` = follow the detected language anyway (earlier 0.8.0) |
| `synthetic_collapse_period` | `True` | Synthetic mode: a fake ending in `.` (de_DE name suffixes such as `B.Eng.`) drops that period when a `.` already follows the entity, so `B.Eng..` becomes `B.Eng.`. `False` keeps both |
| `synthetic_reference_date` | `'2026-01-01'` | Synthetic mode: the ISO date that date fakes (`DATE`, `DATE_OF_BIRTH`) end on and that Faker reads as today (the birthday inside native `SOCIAL_SECURITY_NUMBER`/`TAX_ID` fakes, e.g. `zh_CN`, `fi_FI`, `sv_SE`), so a document gets the same fakes whatever day it is cleaned. `None` uses today's date from the clock |
| `positional_tags` | `('PER', 'LOC', 'ORG', 'MISC')` | Entity types to recognize |
| `ner_confidence_threshold` | `None` | Minimum confidence score (0..1). `None` = 0.85, or the GLiNER threshold in PII mode; an explicit value is always used as given |
| `ner_ensemble_vote` | `'max'` | How the NER models vote on one span: `'max'` keeps it when the best model score reaches `ner_confidence_threshold`, so a weak second model on the same (widened) span never hides a strong hit; `'mean'` averages the scores (0.8.0) |
| `ner_propagate_masked` | `True` | After NER, mask every other case-sensitive whole-word copy of a masked PER/ORG/LOC value in the same document with the same type, so a name masked once is not left visible elsewhere. Values come from the model's own spans (no word lists); values under 2 characters or without a letter are not spread. A wrong tag is repeated too. `False` = 0.8.0 output |
| `ner_propagate_min_score` | `None` | Score (0..1) a span needs before its value is propagated. `None` = the effective NER cut; raise it to spread fewer wrong tags at the cost of more visible repeats |
| `check_replace_iban_cards` | `True` | Before the phone step, mask an IBAN whose mod-97 checksum holds as one `<IBAN>` and a 13-19 digit card number (first digit 2-6, spaces/hyphens allowed) whose Luhn checksum holds as one `<CARD>`. Bad checksums go to the phone/number steps. `False` = 0.8.0 (`NL91ABNA<PHONE>`) |
| `replace_with_iban` | `"<IBAN>"` | Replacement for IBANs |
| `replace_with_cards` | `"<CARD>"` | Replacement for card numbers |
| `check_replace_identifiers` | `False` | Mask identifier tokens: 6+ uppercase letters and digits (any script), 2+ digits and a letter, not glued to another letter or digit (`RBCON68T08`). Off by default: also takes product and reference codes (`RTX4090`, `COVID19`) |
| `replace_with_identifiers` | `"<ID>"` | Replacement for identifier tokens |
| `ner_merge_location_gap` | `0` | Join two neighbouring LOCATION spans into one when the text between them holds at most this many characters (placeholders such as `<NUMBER>` are not counted and are taken in), no line break and no sentence end (`.!?;:` before whitespace, `。！？`). An address like `Elm Street, Apt 5B, Springfield, IL <NUMBER>, USA` becomes one `<LOCATION>` instead of leaving `Apt 5B` and `IL` visible. Format rule only (no word lists). The cost: a short list of places (`Paris, London and Berlin`) also becomes one `<LOCATION>`. `12` or `20` are the measured settings; `0` = off (0.8.0 output) |
| `ner_snap_word_edges` | `True` | A NER span whose start or end falls inside a word moves out to the word edges, for every backend (SentencePiece XLM-R, GLiNER, torch), so `Via della Salut` + `e` or a Bengali name + glued case suffix leaves no letters next to the placeholder. A word is a run of Unicode letters and marks; wide CJK characters never join. The `ner_subword_widen` cap and `ner_subword_widen_same_script` rule apply (an unspaced Thai clause is not swallowed; `ner_subword_widen=0` also turns snapping off). Unicode-property rule only. `False` = 0.8.0 spans |
| `ner_batch_size` | `8` | NER chunks per model forward pass, batched across the documents of a `process_batch` call (must be >= 1). `1` = 0.7.1 call pattern and exact scores, and less memory; which is faster depends on the CPU, so measure |
| `onnx_intra_op_threads` | `None` | ONNX Runtime threads per inference (`None` = ORT default: every physical core, ignoring CPU affinity). Set it to cap CPU use; fewer threads is usually slower |
| `onnx_inter_op_threads` | `None` | ONNX Runtime inter-op threads (`None` = ORT default). Only matters with ORT parallel execution mode, so `onnx_intra_op_threads` is the usual knob |
| `ner_device` | `None` | NER device: `None` = auto (CUDA when available, else CPU; never the Apple GPU), `'cpu'`, `'cuda'`, `'mps'` (Apple GPU for torch NER and GLiNER in torch mode; ONNX stays on CPU; CPU fallback adds `mps_fallback`). GLiClass moves to the GPU only with `'cuda'` |
| `cuda_device_id` | `0` | GPU index for ONNX, torch, GLiNER and GLiClass when on CUDA |
| `onnx_cuda_provider_options` | `None` | Extra `CUDAExecutionProvider` options, e.g. `{"gpu_mem_limit": 8 << 30}`; passed to ONNX Runtime unchecked |
| `ner_chunk_overlap` | `0` | Tokens of neighbouring text each NER window adds on both sides, for documents longer than one window (about 460 tokens with the default models). Finds entities that a chunk boundary would split (`John \| Smith`); costs about `2 * overlap / window` more inference and can change recall either way. `0` = 0.7.1 chunks. Must stay below a quarter of the window |
| `ner_models` | `None` | Language-keyed dict of ONNX model repo IDs |
| `torch_ner_models` | `None` | Language-keyed dict of PyTorch model repo IDs |
| `gliner_model` | `None` | GLiNER model ID (required for gliner/ensemble backends) |
| `gliner_variant` | `'gliner'` | `'gliner'` or `'gliner2'` |
| `gliner_labels` | `('person', 'organization', 'location')` | GLiNER entity labels |
| `gliner_label_map` | `None` | Maps GLiNER labels to NER tags |
| `gliner_threshold` | `None` | GLiNER confidence threshold (0..1). `None` = 0.4, or 0.3 in PII mode |
| `gliner_label_descriptions` | `None` | ZERONER-style: `{label: "description"}` for improved zero-shot accuracy |
| `json_key_cleaning` | `'pii'` | How `process_json` (and `sct-clean --json-input`, MCP `clean_json`, RAG chunk metadata) cleans dict keys: `'pii'` = regex PII steps only (no NER: models tag lone words such as `message` out of context), `'pii_ner'` = PII steps plus NER, `'full'` = the whole config (0.7), `'off'` = keys as written (can carry PII) |
| `fuzzy_date_score_cutoff` | `85` | Fuzzy matching threshold (0-100) for misspelled months |
| `custom_pipeline_steps` | `()` | Tuple of `(text: str) -> str` callables. They run after the built-in fixes, regex replacements and cleanup, and before NER and the `stat_text` steps, so they never see NER tags |

**Language settings**:

| Field | Default | Description |
|-------|---------|-------------|
| `language` | `None` | Pin language (`'en'`, any of the 75 Lingua languages), restrict detection to a set (`('en','nl')`), or `None` for auto-detect. Accepts Lingua names, ISO 639-1, ISO 639-3 codes. |
| `detect_all_languages` | `True` | Auto-detect over all 75 Lingua languages. `False` = 0.7 detector over `supported_languages` only (EN/NL/DE/ES plus extras). A `language` tuple always restricts detection to its entries |
| `language_confidence_warning` | `0.0` | Adds `language_low_confidence` to `metadata['warnings']` when the detected language's confidence is below this (0..1). Off by default: Lingua's top-1 confidence is low for many correct texts (English with names often 0.2-0.5). `0.5` = early 0.8.0. Warning only, never changes routing |
| `language_margin_warning` | `0.15` | Adds `language_low_confidence` when the detected language leads the runner-up by less than this (0..1), i.e. detection is ambiguous (Dutch vs English, Hindi vs Marathi). `0` turns it off. Warning only, never changes routing |
| `legacy_type_names` | `False` | `True` names the phone and IP types `PHONE_NUMBER` / `IP_ADDRESS` (0.7) in reversible placeholders (`<PHONE_NUMBER_0>`), metadata, signals and PII-mode NER tags. `False` = `PHONE` / `IP`, the same name as the `<PHONE>` / `<IP>` tags. See "Type names" |
| `fix_character_width` | `False` | Part of `check_fix_bad_unicode`. `False` keeps full-width CJK punctuation (`，` `！` `：`) and half-width katakana as written, so Chinese text no longer mixes `,` with `。`; full-width ASCII letters and digits (and `＠ ． － ＿ ＋` inside them) still become ASCII, so `０６１２３４５６７８` and `ｊｏｈｎ＠ｅｘａｍｐｌｅ．ｃｏｍ` are still replaced. `True` = ftfy's full width fix (0.7) |
| `ner_subword_widen` | `1.0` | WordPiece NER models (`##` pieces) only: an entity widens to its whole word when the added text is at most this many times the tagged span, so `John Do` + `##e` never leaves `e` visible but a Thai or kana clause with no spaces is not swallowed whole. `None` = always widen (early 0.8.0), `0` = never (0.7). Of the default models only the MULTILINGUAL one (WikiNEural, mBERT) is WordPiece |
| `fold_styled_characters` | `True` | Part of `check_fix_bad_unicode`. Folds math-alphanumeric (`𝐞𝐱𝐚𝐦𝐩𝐥𝐞`, `𝟎𝟔`), full-width and circled (`ⓙⓞⓗⓝ`, only next to ASCII letters/digits, `＠` or URL punctuation) letters and digits to ASCII by their NFKC form, and full-width, small-form and font URL/email punctuation (`： ／ ？ ＝ ＆ ＃ ％ ～ ＠ ． － ＿ ＋`, `﹫ ﹒ ﹕ ﹢ ﬩`, picked by Unicode decomposition) between ASCII letters or digits of any script (after a `scheme://` or `www.`, full-width `： ／ ？ ＝ ＆ ＃ ％ ～` beside any script, and `．` in the host: `https://例え．テスト/`), so styled emails, URLs and phones are replaced with `check_to_ascii_unicode=False`. A lone circled number between or after space-separated digit groups (`06 73 86 88 ㊺`) folds as one of them. Invisible format characters (Unicode category Cf: ZWSP, word joiner, soft hyphen, bidi marks) beside an ASCII letter/digit, any decimal digit or contact punctuation are dropped when the other side is one of those, a letter, a space or the text edge (`06\u00ad12 34`, `jan.de\u200bvries@example.com`); between two other letters (Thai ZWSP, Persian ZWNJ) they stay. The phone step reads every other space (Unicode category Zs: no-break, thin, narrow no-break) as a space and every dash (Pd: hyphen, non-breaking hyphen, en dash) as `-` (`06\u00a012\u00a034\u00a056\u00a078`, `06\u20111234\u20115678`); the text keeps them. Dates read spaces and dashes the same way (`1987\u201105\u201112`), every character whose NFKC form is `.` as a dot (`1990﹒ 1﹒ 15﹒`), and a month-name date's day may be a circled number (`⑮ January 1990`). A circled letter folds beside a Latin letter too (`26 ⓓéⓒⓔⓜⓑⓡⓔ 1990`), and a format character between a letter of any script and a space or the text edge drops (`1990년 6월\u2060 1일`). In a mail address every dash (Pd) is a hyphen and every character whose NFKC form is `.` a dot, local part included (`john.smith\u2011billing@`), and styled forms of the other local-part punctuation fold (`anna＊＠`). In a mail domain or URL host an IDNA dot (`。 ｡`) before an ASCII letter/digit is a dot (`alice@example。com`), and every format character but ZWNJ/ZWJ in a mail domain is dropped (`info@例\u200bえ.jp`). Squared letters (`🄹`) fold like circled ones; inside a mail address or URL run circled and squared letters fold whatever sits beside them (`a@Ŗ🅂Ń.io`). CJK prose punctuation, a lone `①` (also before a number, `① 2024年`), super/subscripts (`m²`, `H₂O`, `nº`) and squared units (`㎏`) stay. `False` = earlier 0.8.0 (full-width letters/digits and `＠ ． － ＿ ＋` only) |
| `ner_subword_widen_same_script` | `True` | WordPiece widening stops where the Unicode script of the letters changes (script from the letter's Unicode name after NFKC; marks, digits and punctuation do not count), so a Latin name glued to kana or Thai (`はDavid Millerで`, `ถึงคุณSarah Lee`) keeps its own span. `False` = widen across scripts (earlier 0.8.0) |
| `extra_languages` | `()` | Additional language names/codes for detection |
| `custom_stopwords` | `None` | `{LANG: frozenset({...})}` custom stopword sets |
| `custom_month_names` | `None` | `{LANG: ('Jan', 'Feb', ...)}` for date detection |
| `custom_deny_terms` | `None` | `{LABEL: ('term', ...)}`: always mask these terms as `<LABEL>` (`{'PROJECT': ('Blue Sky',)}`). Whole words, Unicode-aware: a term matches where no letter, mark or digit of a spaced script touches it (`Acme` in `(Acme)` and `金额ACME元`, not in `Acmes` or `Acme1`); in CJK or Thai the term edge is the word edge (`東京` in `我住在東京都`). Longest term wins an overlap. Labels are upper case (`A-Z 0-9 _`). Runs before every built-in regex step, so the term wins an overlap with an email, phone or number (a term inside an email cuts that email: `info@acme.com` with `acme` gives `info@<ORG>.com`). See [User term lists](REPLACEMENT_MODES.md#user-term-lists). SCT ships no list. Legacy global `CUSTOM_DENY_TERMS` |
| `custom_allow_terms` | `()` | `('term', ...)`: never mask these. A regex or NER detection that equals or lies inside an occurrence of an allowed term (whole-word, same rules) stays as text. Applied last, so it also wins over `custom_deny_terms` and `custom_patterns`. Regex detections are checked by value: a value that also appears outside an allowed occurrence in the document is masked everywhere (`Acme 2024 ... founded 2024` masks both years), so allowing a term never leaks a copy elsewhere. Legacy global `CUSTOM_ALLOW_TERMS` |
| `custom_patterns` | `None` | `{LABEL: r'regex'}`: mask every match as `<LABEL>` (`{'EMPLOYEE_ID': r'EMP-\d{6}'}`). Compiled once with the third-party [`regex`](https://pypi.org/project/regex/) module (Unicode classes such as `\p{L}` work). Runs after `custom_deny_terms`, before the built-in steps. Invalid patterns and patterns that can match the empty string raise `ValueError` at config time (a best-effort probe; empty matches are skipped at run time too). A pattern with nested repeats (`(a+)+`) can backtrack for a very long time (ReDoS); `custom_patterns_timeout` bounds it. Legacy global `CUSTOM_PATTERNS` |
| `custom_terms_case_sensitive` | `False` | Whether `custom_deny_terms` and `custom_allow_terms` match case-sensitively. `False`: `acme` matches `ACME`. Terms of both lists are normalised like the text (unicode fix, ASCII, emoji steps) before matching. Legacy global `CUSTOM_TERMS_CASE_SENSITIVE` |
| `custom_patterns_timeout` | `2.0` | Seconds each `custom_patterns` regex may spend on one document (the `regex` module's `timeout`), so a pattern that backtracks badly cannot hang `process`, `process_batch`, the streaming redactors or `process_json`. `None` = no limit. Legacy global `CUSTOM_PATTERNS_TIMEOUT` |
| `custom_patterns_on_timeout` | `'raise'` | What a timeout does: `'raise'` raises `sct.CustomPatternTimeout` (a `TimeoutError`; the message names the label, never the text); `'warn'` logs a warning, adds `custom_pattern_timeout` to `metadata['warnings']` and processes that document without the pattern (its other rules still run). Legacy global `CUSTOM_PATTERNS_ON_TIMEOUT` |
| `numbers_keep_currency_code_amounts` | `True` | `'identifiers'` mode: a whole digit run right before or after an uppercase ISO 4217 code with at most one space between (`1250 EUR`, `EUR 1250`, `USD 1,250`) is an amount and stays. ID shapes next to a code are still masked whole: SSN-like, slash, dot and spaced runs (`SSN 123-45-6789 USD`, `123456/7890 EUR`, `EUR 123 45 6789`) and letter-prefix IDs. Codes are the set that already keeps `USD1000000` out of letter-prefix IDs; lower case (`eur`) and codes inside a word (`EURO`, `PEUR`) do not count. IBANs and cards next to a code are still masked by their own step. `False`: masked like any ID. Legacy global `NUMBERS_KEEP_CURRENCY_CODE_AMOUNTS` |


## Legacy configuration


```python
from sct import sct, config

# Customize settings via module-level variables
config.CHECK_NER_PROCESS = True
config.NER_CONFIDENCE_THRESHOLD = 0.85
config.POSITIONAL_TAGS = ['PER', 'LOC', 'ORG']
config.REPLACE_WITH_URL = "<URL>"
config.REPLACE_WITH_EMAIL = "<EMAIL>"
config.LANGUAGE = "ENGLISH"

# Initialize (reads from module-level config)
cleaner = sct.TextCleaner()
```

Every `TextCleanerConfig` field has a global with its upper-case name (`ner_backend` is `config.NER_BACKEND`), except the computed `supported_languages` and the deprecated `ner_models_list`, whose global `NER_MODELS_LIST` feeds `ner_models`; a dict in `NER_MODELS` wins over it.

> **Note**: The legacy module-level configuration is not thread-safe. For concurrent processing, use `TextCleanerConfig` instead.

