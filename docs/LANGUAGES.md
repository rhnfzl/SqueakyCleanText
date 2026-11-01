# Languages

| Part | Languages |
|---|---|
| Language detection ([Lingua](https://github.com/pemistahl/lingua-py)) | all 75 languages below (`detect_all_languages=True`) |
| NER models of their own | English, Dutch, German, Spanish |
| Multilingual NER model (WikiNEural) | French, Portuguese, Italian, and the fallback for every other language |
| Stopwords (`stat_text`) | English, Dutch, German, Spanish; other languages use English |
| Month names (dates) | English, Dutch, German, Spanish, French, Portuguese, Italian, plus `custom_month_names` |

A language with no NER model of its own runs the multilingual and English models and gets
the `ner_fallback_models` warning. `ner_backend='gliner'` is language independent, so any
pinned language works with it. To bring your own model for a language, set
`ner_models={"HINDI": "your/hindi-ner-onnx", ...}`. Adding stopwords, month names and NER
models: [ADDING_LANGUAGES.md](ADDING_LANGUAGES.md). How the default ensemble scores in 11
languages: [MULTILINGUAL_EVAL.md](MULTILINGUAL_EVAL.md).

Every language option takes the Lingua name (`'ENGLISH'`), ISO 639-1 (`'en'`) or ISO 639-3
(`'eng'`). `language='de'` pins one language, `language=('en', 'nl')` limits detection to a
set; see [CONFIG.md](CONFIG.md#languages).

## Detected languages (75)

| Lingua name | ISO 639-1 | ISO 639-3 |
|---|---|---|
| AFRIKAANS | `af` | `afr` |
| ALBANIAN | `sq` | `sqi` |
| ARABIC | `ar` | `ara` |
| ARMENIAN | `hy` | `hye` |
| AZERBAIJANI | `az` | `aze` |
| BASQUE | `eu` | `eus` |
| BELARUSIAN | `be` | `bel` |
| BENGALI | `bn` | `ben` |
| BOKMAL | `nb` | `nob` |
| BOSNIAN | `bs` | `bos` |
| BULGARIAN | `bg` | `bul` |
| CATALAN | `ca` | `cat` |
| CHINESE | `zh` | `zho` |
| CROATIAN | `hr` | `hrv` |
| CZECH | `cs` | `ces` |
| DANISH | `da` | `dan` |
| DUTCH | `nl` | `nld` |
| ENGLISH | `en` | `eng` |
| ESPERANTO | `eo` | `epo` |
| ESTONIAN | `et` | `est` |
| FINNISH | `fi` | `fin` |
| FRENCH | `fr` | `fra` |
| GANDA | `lg` | `lug` |
| GEORGIAN | `ka` | `kat` |
| GERMAN | `de` | `deu` |
| GREEK | `el` | `ell` |
| GUJARATI | `gu` | `guj` |
| HEBREW | `he` | `heb` |
| HINDI | `hi` | `hin` |
| HUNGARIAN | `hu` | `hun` |
| ICELANDIC | `is` | `isl` |
| INDONESIAN | `id` | `ind` |
| IRISH | `ga` | `gle` |
| ITALIAN | `it` | `ita` |
| JAPANESE | `ja` | `jpn` |
| KAZAKH | `kk` | `kaz` |
| KOREAN | `ko` | `kor` |
| LATIN | `la` | `lat` |
| LATVIAN | `lv` | `lav` |
| LITHUANIAN | `lt` | `lit` |
| MACEDONIAN | `mk` | `mkd` |
| MALAY | `ms` | `msa` |
| MAORI | `mi` | `mri` |
| MARATHI | `mr` | `mar` |
| MONGOLIAN | `mn` | `mon` |
| NYNORSK | `nn` | `nno` |
| PERSIAN | `fa` | `fas` |
| POLISH | `pl` | `pol` |
| PORTUGUESE | `pt` | `por` |
| PUNJABI | `pa` | `pan` |
| ROMANIAN | `ro` | `ron` |
| RUSSIAN | `ru` | `rus` |
| SERBIAN | `sr` | `srp` |
| SHONA | `sn` | `sna` |
| SLOVAK | `sk` | `slk` |
| SLOVENE | `sl` | `slv` |
| SOMALI | `so` | `som` |
| SOTHO | `st` | `sot` |
| SPANISH | `es` | `spa` |
| SWAHILI | `sw` | `swa` |
| SWEDISH | `sv` | `swe` |
| TAGALOG | `tl` | `tgl` |
| TAMIL | `ta` | `tam` |
| TELUGU | `te` | `tel` |
| THAI | `th` | `tha` |
| TSONGA | `ts` | `tso` |
| TSWANA | `tn` | `tsn` |
| TURKISH | `tr` | `tur` |
| UKRAINIAN | `uk` | `ukr` |
| URDU | `ur` | `urd` |
| VIETNAMESE | `vi` | `vie` |
| WELSH | `cy` | `cym` |
| XHOSA | `xh` | `xho` |
| YORUBA | `yo` | `yor` |
| ZULU | `zu` | `zul` |
