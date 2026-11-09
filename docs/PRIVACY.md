# Privacy engine

Applications can apply versioned policy rules and inspect each decision.
This makes privacy behavior testable across services, datasets, and model upgrades.

```python
from sct import EntityRule, PrivacyPolicy, TextCleaner, TextCleanerConfig

policy = PrivacyPolicy(
    name="support-export",
    version="1",
    rules={
        "PERSON": EntityRule(action="redact", min_score=0.8),
        "ORGANISATION": EntityRule(action="keep"),
    },
    allowlist=frozenset({"Jane Public"}),
)

cleaner = TextCleaner(
    cfg=TextCleanerConfig(ner_mode="pii"),
    privacy_policy=policy,
    include_entities=True,
)
result = cleaner.process("Contact Maria Chen for help.")

print(result.metadata["findings"])
print(result.metadata["provenance"])
```

`metadata["findings"]` lists every decision. `entity_counts`, `pii_count` and
`sensitive` count only `placeholder` and `redact` findings, so an entity the
policy kept, allowed or ignored is not reported as PII. OCR redaction covers a
region when its text had anything replaced, dates and years included, and leaves
regions holding only exempted entities visible.

Finding offsets (`start`, `end`) and NER entries in the reversible map are
document-relative positions in the post-regex text: the text after the regex
stages replaced URLs, emails, phones and the like. They are not offsets into the
raw input.

Reversible output is pseudonymized personal data. Anyone with the token map can
restore the source values. Keep token maps outside logs and cleaned datasets.

Use a `TokenStore` implementation to keep maps behind opaque references:

```python
from sct import InMemoryTokenStore, TextCleaner, TextCleanerConfig

store = InMemoryTokenStore()
cleaner = TextCleaner(
    cfg=TextCleanerConfig(replacement_mode="reversible"),
    token_store=store,
)
result = cleaner.process("Maria lives in Utrecht.")
reference = result.metadata["anon_map_reference"]
```

`InMemoryTokenStore` supports tests and short sessions.
For encrypted maps across restarts, install the optional `token-store` extra and use
[`EncryptedSQLiteTokenStore`](TOKEN_STORES.md). The application owns trusted scope assignment,
authorization, keys, audit, backups, and retention.
Deletion and expiry are logical operations, not secure erasure of journals, WAL files, backups, or media.
The local reference integration requires an independent security-focused review before production readiness is claimed.

## Re-identification risk

Corpus owners can call `assess_reidentification_risk` to measure k-anonymity
for selected quasi-identifiers. Reports contain counts only, not source values.
