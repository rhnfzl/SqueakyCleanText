---
edit_url: https://github.com/rhnfzl/SqueakyCleanText/edit/main/README.md
---
<!-- Rendered from the Limits section of the repository README. -->
# Limits

--8<-- "README.md:limits"

## Known limits of 0.8.3

Details in [CHANGELOG.md](https://github.com/rhnfzl/SqueakyCleanText/blob/main/CHANGELOG.md).

- Dates: only full dates become one `<DATE>`. `1990年1月`, `1月15日`, Japanese era years and `1990.1.15.` do not; the year and number steps still change them (`1990年1月` becomes `<YEAR>年1月`, or `<YEAR>年<NUMBER>月` with `replace_numbers_mode='all'`). Thai month names need `custom_month_names`.
- In ensembles with GLiNER, windows are sized for the token-classification models, not for GLiNER's own word limit. Torch models and GLiNER keep one lock per model, so they do not run in parallel across threads.
- Only `process_batch` is faster; `process` of one document is not batched across documents.
- `ner_chunk_overlap` (off by default) can still cut an entity longer than the overlap that starts at the very end of a chunk.
- Number masking (`replace_numbers_mode='identifiers'`, the default) is a format rule: it cannot tell a bare 4+ digit amount with no sign, separator or currency code (`price 1250`) from an ID, and a value joined into a code (`INV-2023-107683`) stays visible unless you mask it yourself. `phone_reject_embedded` leaves a phone glued to a Latin word (`Tel0612345678`) unmasked.
- `06-12345678` becomes `06-<PHONE>`. Adversarial strings such as `+000000000(000)0` can leave `(000)0` after phone replacement.
- ORGANISATION is the weakest NER label (0.75 recall in the multilingual eval); Bengali and Tamil name spans can be a few characters off.
- A URL path that switches from Latin letters straight into a script written without spaces is replaced only up to the switch (`https://x.org/a東京secret` becomes `<URL>東京secret`).
- Score B's quality head was tested with a small ONNX graph only; GLiNER PII mode was not run end to end for this release.
- An unpinned NER model makes one Hub request per worker when the parent never loaded it before forking; pin `ner_model_revisions` or set `HF_HUB_OFFLINE=1`.
- `WindowedStreamingRedactor` raises `BufferError` when the buffer passes `max_buffer_chars` (default 4096) with no safe whitespace split; raise the limit or use `BufferedStreamingRedactor`. A date or IBAN longer than `overlap_chars` (default 256) that is still unfinished at the buffer end can be split and leak; it refuses `custom_allow_terms`.
