# Rules for code changes

## No keyword-based fixes

Never fix a detection or model mistake with a list of words: no allow lists, deny lists,
ignore lists or per-word exceptions, in any language. Example: the German model tags `Büro`
as an organisation. Adding `Büro` to a list is not a fix.

Fix model mistakes with model-driven rules that work for every language: confidence
thresholds, agreement between ensemble models, span scores. Measure each rule on the
evaluation sets (`evaluation/`) and ship it only when the numbers improve. If nothing
helps, document it as a known limit with the numbers.

Allowed: regex patterns for formats (emails, phones, dates, IPs, URLs) and the existing
user-facing features that take word data (stopword removal, custom date formats).

## Every behaviour is configurable

Every behaviour is a safe default plus a config option (`TextCleanerConfig` field and
legacy module global). Never force one behaviour.
