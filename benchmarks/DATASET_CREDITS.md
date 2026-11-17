# Dataset credits

SqueakyCleanText benchmark pages show scores measured on the open datasets below.
Their licences ask for credit wherever those scores appear, so every page and chart that shows them links here.

For each dataset: title, creators, source, licence, and the citation its card asks for.

**Changes we made.** We do not redistribute any of this data.
The loaders in [`benchmarks/datasets/`](https://github.com/rhnfzl/SqueakyCleanText/tree/main/benchmarks/datasets) download it at a pinned revision,
take a seeded sample, and map each dataset's labels to SCT types.
We publish only aggregate scores, plus record ids where the licence allows it.
The scores are our own measurements. They do not mean the dataset creators endorse SqueakyCleanText.

## Creative Commons Attribution (CC BY)

| Dataset (loader) | Creators | Source | Licence | Cite |
| --- | --- | --- | --- | --- |
| OpenPII 1.5M: Multilingual PII Masking with Asia Pacific Extension (`openpii`) | Ai4Privacy / Ai Suisse SA. © 2026 Ai Suisse SA | [ai4privacy/pii-masking-openpii-1.5m](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Ai4Privacy (2026), OpenPII 1.5M, Hugging Face |
| Nemotron-PII: Synthesized Data for Privacy-Preserving AI (`nemotron`) | NVIDIA: Amy Steier, Andre Manoel, Alexa Haushalter, Maarten Van Segbroeck | [nvidia/Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Steier et al. (2025), Nemotron-PII, NVIDIA |
| IAHLT Named Entities Dataset, Arabic subset (`iahlt`) | The Israeli Association of Human Language Technologies ([IAHLT](https://www.iahlt.org)) | [HebArabNlpProject/arabic-iahlt-NER](https://huggingface.co/datasets/HebArabNlpProject/arabic-iahlt-NER) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | |
| Arabic PII evaluation set (synthetic) (`bayanat`) | Bayanat Labs ([bayanatlabs.com](https://bayanatlabs.com)) | [Bayanat-Labs/arabic-pii-eval](https://huggingface.co/datasets/Bayanat-Labs/arabic-pii-eval) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | "Arabic PII evaluation set, Bayanat Labs (https://bayanatlabs.com), CC BY 4.0" |
| Thai NER 2.2 (`thainer`) | Wannaphong Phatthiyaphaibun (PyThaiNLP) | [pythainlp/thainer-corpus-v2.2](https://huggingface.co/datasets/pythainlp/thainer-corpus-v2.2) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) (Hugging Face card); [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([Zenodo record](https://doi.org/10.5281/zenodo.10795907)) | Phatthiyaphaibun (2024), Thai NER 2.2, Zenodo, doi:10.5281/zenodo.10795907 |
| Thai PII Synthetic NER Dataset (`thai_pii`) | DuckerMaster | [DuckerMaster/Thai-PII-Synthetic](https://huggingface.co/datasets/DuckerMaster/Thai-PII-Synthetic) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | DuckerMaster (2026), Thai PII Synthetic NER Dataset |
| MultiCoNER v2 (`multiconer`) | Besnik Fetahu, Zhiyu Chen, Sudipta Kar, Oleg Rokhlenko, Shervin Malmasi | [MultiCoNER/multiconer_v2](https://huggingface.co/datasets/MultiCoNER/multiconer_v2) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Fetahu et al. (2023), MultiCoNER v2: a Large Multilingual dataset for Fine-grained and Noisy Named Entity Recognition; SemEval-2023 Task 2 |
| Wikipediaを用いた日本語の固有表現抽出データセット (Japanese NER from Wikipedia) (`stockmark`) | Stockmark Inc.; text from Japanese Wikipedia | [stockmark/ner-wikipedia-dataset](https://huggingface.co/datasets/stockmark/ner-wikipedia-dataset) | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) | 近江崇宏 (Takahiro Omi), NLP2021 |
| DialogPII: Multilingual Synthetic Dialogs and Transcripts Annotated for PII (`dialogpii`) | Roland Roller, Vera Czehmann, Derya Erman, Luke Flanagan, Ibrahim Baroud, Frédéric Blain, Viviana Cotik, Eletta Giusto, Akhil Juneja, Mariana Neves, Maria Słowińska, Christine Hovhannisyan, Aaron Louis Eidt, Lisa Raithel, Sebastian Möller, Maija Poikela | [Zenodo 20863452](https://doi.org/10.5281/zenodo.20863452) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Roller et al. (2026), DialogPII v1.0, Zenodo, doi:10.5281/zenodo.20863452 |
| KDPII: Korean Dialogic Dataset for PII De-identification (`kdpii`) | Yonsei University HamSaeM Kim's Lab; TSCIENTIFIC Co., Ltd; Hansaem Kim and co-authors, as listed on the Zenodo record | [Zenodo 10968609](https://doi.org/10.5281/zenodo.10968609) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Fei et al. (2024), KDPII: A New Korean Dialogic Dataset for the Deidentification of Personally Identifiable Information, IEEE Access 12:135626-135641 |
| EuroPII benchmark (`europii`) | Since AI | [sinceaihq/europii-bench](https://huggingface.co/datasets/sinceaihq/europii-bench) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Since AI (2026), EuroPII benchmark |
| Greek Clinical De-identification Benchmark (`greek_deid`) | Filippos-Paschalis Rorris, Nikoleta Stanitsa, Tatiana Rorri, Foteini Rorri, Constantine N. Antonopoulos | [Zenodo 22824895](https://doi.org/10.5281/zenodo.22824895) (mirror [prorris/greek-deid-benchmark](https://huggingface.co/datasets/prorris/greek-deid-benchmark)) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Rorris et al. (2026), v1.0.0, Zenodo, doi:10.5281/zenodo.22824895 |
| MedDeID English and Dutch synthetic benchmarks (`meddeid`) | Stig Hellemans, Tom Stroobants, Elyne Scheurwegs, Pieter Meysman, Philippe Jorens, Kris Laukens (University of Antwerp, UZA) | [English](https://huggingface.co/datasets/stighellemans/meddeid-english-synthetic-benchmark), [Dutch](https://huggingface.co/datasets/stighellemans/meddeid-dutch-synthetic-benchmark) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Hellemans et al. (2026), Zenodo doi:10.5281/zenodo.22689857 and doi:10.5281/zenodo.22689856; arXiv:2609.10049 |
| NeonRedact-TR Bench (`neonredact`) | Aykan Kömürcü, NeonDijital | [neondijital/neonredact-tr-bench](https://huggingface.co/datasets/neondijital/neonredact-tr-bench) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Kömürcü (2026), NeonRedact-TR Bench: An Evaluation Set for Turkish PII Detection |
| newmean PII dataset, public test (`newmean`) | [newmean.ai](https://newmean.ai) | [newmean-ai/pii-dataset](https://huggingface.co/datasets/newmean-ai/pii-dataset) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial evaluation) | Made by newmean.ai |
| pl-pii-bench v1.0.0 (`pl_pii`) | Anonimator.pl. © 2026 Anonimator | [pl-pii-bench/pl-pii-bench](https://huggingface.co/datasets/pl-pii-bench/pl-pii-bench) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (data) | Anonimator.pl (2026), pl-pii-bench: An Open Benchmark for Polish PII Detection and Anonymization |
| RedactionBench (`redactionbench`) | RedactionBench authors | [RedactionBench/RedactionBench](https://huggingface.co/datasets/RedactionBench/RedactionBench) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); 49 source documents are all rights reserved, so aggregate scores only | |
| PrivacyBench v2.5 (`tonic`) | Tonic AI | [TonicAI/Privacy-Bench](https://huggingface.co/datasets/TonicAI/Privacy-Bench) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Tonic AI (2026), PrivacyBench: A Benchmark for Synthesizing Personal Workplace Data |

## Other attribution licences

| Dataset (loader) | Creators | Source | Licence | Cite |
| --- | --- | --- | --- | --- |
| FineWeb2 (`fineweb2`, speed runs only) | Hugging Face: Guilherme Penedo et al. | [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/); also [Common Crawl terms](https://commoncrawl.org/terms-of-use) | Penedo et al. (2025), FineWeb2, arXiv:2506.20920 |
| WiLI-2018: Wikipedia Language Identification database (`wili`) | Martin Thoma | [Zenodo 841984](https://doi.org/10.5281/zenodo.841984) (mirror [MartinThoma/wili_2018](https://huggingface.co/datasets/MartinThoma/wili_2018)) | [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/) | Thoma (2018), WiLI-2018, doi:10.5281/zenodo.841984 |
| Naamapadam (`naamapadam`) | AI4Bharat: Arnav Mhaske, Harshit Kedia, Sumanth Doddapaneni, Mitesh M. Khapra, Pratyush Kumar, Rudra Murthy, Anoop Kunchukuttan | [ai4bharat/naamapadam](https://huggingface.co/datasets/ai4bharat/naamapadam) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) per card; its loader script says CC BY-NC 4.0, so we credit it | Mhaske et al. (2022), Naamapadam, arXiv:2212.10168 |

The other benchmark datasets use licences without a credit term on scores (Apache-2.0, MIT, the Ai4Privacy licence, REDACT terms, NewGenLTU Open RAIL-D, PhysioNet).
[`benchmarks/README.md`](https://github.com/rhnfzl/SqueakyCleanText/blob/main/benchmarks/README.md) lists every dataset with its licence and pinned revision.
