# Sustainable Public Procurement NLP Project Playbook

This repository follows a practical architecture for classifying procurement PDFs by the strength of sustainability requirements.

## One-Sentence Architecture

```text
Raw PDF
-> text extraction or OCR
-> basic cleaning while preserving evidence
-> document and evidence labeling
-> split complete documents into train/validation/test
-> train baseline NLP classifier
-> evaluate on held-out documents
-> export prediction + confidence + evidence
```

## Primary Prediction Unit

```text
One complete procurement document = one primary training sample
```

Example:

| document_id | source_file | final_level |
|---|---|---:|
| RFP_001 | BC_Hydro_RFP.pdf | 3 |
| RFP_002 | Toronto_RFP.pdf | 5 |
| RFP_003 | Oakville_RFP.pdf | 2 |

## Evidence Layer

To support explainability, store evidence spans separately:

| document_id | page_number | evidence_text | local_level | evidence_type |
|---|---:|---|---:|---|
| RFP_002 | 31 | Supplier shall provide valid ISO 14001 certification. | 5 | mandatory certification |
| RFP_004 | 22 | Sustainability accounts for 15% of total score. | 4 | weighted criteria |
| RFP_003 | 7 | The City supports green procurement practices. | 2 | mention only |

## Final-Level Rule

Default rule:

```text
Final document level = the highest applicable and enforceable sustainability requirement level evidenced in the document.
```

This must be documented in the labeling codebook to avoid inconsistent labels.

## Data Lineage

Recommended layout:

```text
data/
├── raw_pdfs/
├── extracted_text/
├── cleaned_text/
├── annotations/
├── interim/
└── processed/
```

Each derived record should retain:

- `document_id`
- `source_file`
- `page_number`
- `extraction_method`
- `processing_version`

## Cleaning Principles

Readable cleaning should:

- remove repeated headers and footers;
- remove page numbers and watermark noise;
- normalize whitespace;
- preserve numbers, percentages, negation, modal verbs, and certifications.

Do not blindly remove:

- `must`
- `shall`
- `may`
- `not`
- `mandatory`
- `15%`
- `ISO 14001`

## Modeling Strategy

Recommended first model:

```text
basic-cleaned document text
-> TF-IDF (unigrams + bigrams)
-> Logistic Regression
```

Why:

- works with small datasets;
- interpretable;
- easy to debug;
- good baseline for later semantic experiments.

Semantic experiments can include:

- sentence embeddings + lightweight classifier
- chunk-level aggregation
- later BERT fine-tuning after labels mature

## Leakage Prevention

Correct order:

```text
split complete documents first
-> fit vectorizer on training data only
-> train model
-> evaluate on held-out documents
```

Never:

- split pages from the same PDF across train/test
- fit TF-IDF vocabulary on the full dataset
- use test documents to choose features

## Core Evaluation

Report:

- macro F1
- per-class precision/recall/F1
- confusion matrix
- class distribution
- error analysis with likely cause

## Quality Gate

Do not claim success unless:

- raw PDFs are traceable
- labels are versioned
- train/test split is at document level
- preprocessing leakage is prevented
- evidence is stored with page references
- results are reported honestly

