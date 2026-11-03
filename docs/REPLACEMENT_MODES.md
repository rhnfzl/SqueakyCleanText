# Replacement modes and document classification

`replacement_mode` picks what a found PII value becomes: `'placeholder'` (default, a `<TAG>`),
`'synthetic'` (a fake value) or `'reversible'` (an indexed placeholder plus a map to restore it).

## Synthetic Replacement

Replace detected entities with realistic fake values (via [Faker](https://faker.readthedocs.io/)) instead of `<TAG>` placeholder tokens:

```python
from sct import TextCleaner, TextCleanerConfig

cfg = TextCleanerConfig(replacement_mode='synthetic')  # pip install squeakycleantext[synthetic]
cleaner = TextCleaner(cfg=cfg)
print(cleaner.process("Contact John Smith at john.smith@company.com or +1-555-0123").lm_text)
# -> Contact Jamie Douglas at akemp@example.net or 406-398-8659x17221
```

The same entity always maps to the same fake value within a document, and the same document
always gets the same fakes (`synthetic_seed`, `synthetic_per_document`). The exact fakes depend
on the installed Faker version (this output: Faker 40.44). Works with `ner_mode='pii'` too.

> **Note**: Synthetic replacement preserves data utility for downstream ML tasks but is NOT GDPR-compliant anonymization. Same-document consistency is maintained (same entity text always maps to the same fake value).

## Reversible Anonymization

Replace entities with indexed placeholders (`<PERSON_0>`, `<LOCATION_1>`) and get a mapping for round-trip deanonymization:

```python
import json
from sct import TextCleaner, TextCleanerConfig

cleaner = TextCleaner(cfg=TextCleanerConfig(replacement_mode='reversible'))
result = cleaner.process("John Smith works at Google in London.")
print(result.lm_text)  # -> <PERSON_0> works at <ORGANISATION_0> in <LOCATION_0>.

# Access the anonymization map via metadata
anon_map = result.metadata['anon_map']
print(anon_map.deanonymize(result.lm_text))  # -> John Smith works at Google in London.

# Serialize the map for storage
stored = json.dumps(anon_map.to_dict())
```


## User term lists

SCT never ships a word list, but you can pass your own. Every mode uses them: placeholder, reversible,
synthetic, `process_batch`, the streaming redactors and `process_json`.

```python
cfg = TextCleanerConfig(
    custom_deny_terms={'PROJECT': ('Blue Sky', '蓝天计划')},   # always masked as <PROJECT>
    custom_patterns={'EMPLOYEE_ID': r'EMP-\d{6}'},            # your own regex, masked as <EMPLOYEE_ID>
    custom_allow_terms=('support@acme.com',),                 # never masked
)
TextCleaner(cfg).process("Blue Sky: ask support@acme.com or bob@x.com, EMP-123456").lm_text
# -> "<PROJECT>: ask support@acme.com or <EMAIL>, <EMPLOYEE_ID>"
```

- Order: deny terms, then your patterns, then the built-in regex steps and NER; allow terms are applied last
  and win over everything, also over your own deny terms and patterns.
- Placeholder mode writes `<LABEL>`; reversible mode writes `<LABEL_0>` and records the original in the map,
  so `deanonymize` restores it; synthetic mode writes a Faker value when the label has one (`PERSON`, `CITY`,
  `EMAIL`, ...; see `FAKER_GENERATORS` in `sct/utils/synthetic.py`) and `<LABEL>` otherwise.
- `metadata['regex_counts']` and `signals` count each label (count and characters, never values); user labels
  count as PII in `signals['pii_types']`.
- Whole-word and case rules: see `custom_deny_terms` and `custom_terms_case_sensitive` in
  [CONFIG.md](CONFIG.md). The browser manifest does not export these lists (server-side only).
- Time limit: each pattern gets `custom_patterns_timeout` seconds per document (default `2.0`), so a pattern
  that backtracks badly (`(a|aa)+$`, ReDoS) cannot hang processing. On a timeout SCT raises
  `CustomPatternTimeout` naming the label (never the text), or with `custom_patterns_on_timeout='warn'` logs
  it, adds `custom_pattern_timeout` to `metadata['warnings']` and skips that pattern for that document.

## Document Classification (GLiClass)

Classify documents before processing using zero-shot classification with [GLiClass](https://github.com/Knowledgator/GLiClass):

```python
from sct import TextCleaner, TextCleanerConfig

cfg = TextCleanerConfig(
    check_classify_document=True,
    gliclass_labels=('email', 'code', 'legal', 'medical'),
    # gliclass_model defaults to 'knowledgator/gliclass-edge-v3.0' (32.7M params)
)

cleaner = TextCleaner(cfg=cfg)  # pip install squeakycleantext[classify]
result = cleaner.process("Dear Sir, please find attached the contract...")

# Classification results in metadata
print(result.metadata['classes'])
# [{"label": "email", "score": 0.92}]
```

The default `gliclass_classification_type='single-label'` returns only the top label.
Set `gliclass_classification_type='multi-label'` to get every label above the threshold.

GLiClass runs in torch only. The gliclass package has no ONNX loader, so `gliclass_onnx=True`
raises `ValueError`. Configs from before 0.7 that set it must drop it.
