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
# [{"label": "email", "score": 0.92}, {"label": "legal", "score": 0.78}]
```

GLiClass runs in torch only. The gliclass package has no ONNX loader, so `gliclass_onnx=True`
raises `ValueError`. Configs from before 0.7 that set it must drop it.
