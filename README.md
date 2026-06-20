# Sustainable Public Procurement NLP Pipeline

Applied NLP project for analyzing public procurement documents and classifying the strength of sustainability requirements in RFP/RFQ tender files.

## Project Snapshot

This project grew out of a client-facing academic-industry prototype connected to the George Weston Ltd. Centre for Sustainable Supply Chains at York University.

The business goal was to help public institutions answer questions like:

- How often does sustainability appear in procurement documents?
- How strong are those sustainability requirements?
- Which clauses are only aspirational, and which are mandatory?
- How can procurement teams benchmark themselves and improve future tender drafting?

The practical prototype focused on `CanadaBuys` as the main document source and combined:

- web scraping / document collection
- PDF download automation
- PDF text extraction
- text cleaning
- manual sustainability labeling
- NLP-based classification
- Excel / dashboard-oriented outputs

## What I Did

This repository is designed to make it obvious what I actually worked on.

I contributed to an end-to-end workflow involving:

- collecting procurement documents from public sources
- downloading and organizing PDFs
- extracting machine-readable text from procurement files
- cleaning and normalizing text for NLP use
- designing and applying a five-level sustainability scale
- manually labeling sustainability-relevant content
- experimenting with BERT-based semantic features
- building a supervised text classification prototype
- preparing outputs for reporting and dashboard use
- documenting the pipeline for future handoff

## Skills Demonstrated

This project demonstrates practical experience with:

- `Python`
- `NLP preprocessing`
- `PDF parsing`
- `web scraping / document collection`
- `regex-based text cleaning`
- `feature engineering`
- `TF-IDF`
- `BERT / embeddings`
- `supervised multiclass classification`
- `annotation design`
- `data pipeline thinking`
- `model explainability`
- `handoff / project documentation`

## Business Output

The core classification task is:

```text
Procurement document -> sustainability commitment level (1 to 5)
```

### Sustainability Scale

| Level | Meaning |
|---:|---|
| 1 | No sustainability content |
| 2 | Sustainability mentioned, but not operationalized |
| 3 | Certification, disclosure, or sustainability evidence requested |
| 4 | Sustainability explicitly evaluated with score / weighting / criteria |
| 5 | Sustainability is mandatory / pass-fail / condition of award |

## Why This Problem Is Interesting

This is not just a generic text classification problem.

The hard parts are:

- procurement PDFs are messy and inconsistent
- important evidence may be buried in long documents
- sustainability language can be weak, strong, weighted, or mandatory
- labeling policy matters as much as model choice
- explainability is critical because users need to know *why* a file was classified at a certain level

## End-to-End Pipeline

```text
Public procurement source
-> scrape metadata / collect links
-> download PDFs
-> extract text from native PDFs
-> flag encrypted / scanned / OCR cases
-> basic readable cleaning
-> document-level and evidence-level annotation
-> baseline NLP model
-> predicted level + confidence + evidence export
-> dashboard / reporting output
```

## Architecture Choice

The most important modeling decision is:

```text
One complete procurement document = one primary training sample
```

That keeps the task aligned with the business outcome:

```text
Full RFP -> one final sustainability level
```

To support explainability, the project also tracks evidence spans:

```text
document_id, page_number, evidence_text, local_level, evidence_type
```

## Historical Prototype Context

From the original project deliverables:

- `5,657` procurement-related documents were scraped
- `3,087` were English
- `2,546` were French
- the team manually reviewed and labeled a subset of RFP documents
- dashboard-oriented outputs were handed over for future continuation

That matters because this was not a toy exercise. It started as a real prototype with handoff expectations, imperfect data, and evolving scope.

## Repository Structure

```text
.
├── README.md
├── PROJECT_PLAYBOOK.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── baseline_tfidf.yaml
├── data/
│   └── annotations/
│       ├── document_labels.template.csv
│       ├── evidence_spans.template.csv
│       ├── label_codebook.md
│       └── sample_training_data.csv
├── docs/
│   ├── FUTURE_IMPROVEMENTS.md
│   ├── INTERVIEW_GUIDE.md
│   ├── PROJECT_CONTEXT.md
│   └── REPO_EVOLUTION.md
├── src/
│   ├── cleaning/
│   ├── ingestion/
│   ├── labeling/
│   ├── models/
│   └── utils/
└── tests/
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the minimal baseline demo

This repo includes a small synthetic training dataset purely so the project can run end-to-end as a demo.

```bash
python src/models/train_baseline.py \
  --dataset data/annotations/sample_training_data.csv \
  --config configs/baseline_tfidf.yaml \
  --metrics-output /tmp/sppp_baseline_metrics.json
```

### 3. Run tests

```bash
pytest -q
```

## Current State

This repository is intentionally honest about maturity.

What already exists here:

- a clear business framing
- annotation templates and label codebook
- a baseline project structure
- ingestion / cleaning / training skeletons
- a minimal runnable demo
- documentation for interview and future handoff

What is still prototype-level:

- the real historical notebooks are not yet fully migrated into `src/`
- the production-quality annotation tables are not yet versioned here
- OCR/table extraction is not yet implemented end-to-end
- model evaluation on the original corpus is not yet reproduced in this repo

## How I Would Explain Model Choice

The original prototype explored BERT-based features because sustainability language is contextual.

However, for a cleaner and more defensible repository, the recommended modeling order is:

1. `TF-IDF + Logistic Regression` as the first reproducible baseline
2. `TF-IDF + procurement lexicon features` for stronger interpretability
3. `sentence embeddings + lightweight classifier` for semantic improvement
4. `fine-tuned BERT` only after labels, chunking, and evaluation design are stable

Why not fine-tune BERT first?

- the labeled dataset is relatively small
- procurement documents are long and need chunking
- explainability matters
- a simpler baseline is easier to validate and debug

## Future Improvements

The future roadmap is documented in [docs/FUTURE_IMPROVEMENTS.md](docs/FUTURE_IMPROVEMENTS.md).

High-priority next steps:

- migrate the original notebook logic into reproducible scripts
- separate document-level labels from evidence-level labels
- preserve page-level source linkage
- build a proper TF-IDF benchmark on real labeled data
- compare that benchmark against sentence embeddings
- only then decide whether fine-tuning BERT is justified

## Best Way To Read This Repository

If you want the shortest path:

1. Read this README
2. Read [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md)
3. Read [docs/FUTURE_IMPROVEMENTS.md](docs/FUTURE_IMPROVEMENTS.md)
4. Review `src/` for the pipeline skeleton

## Suggested Repository Title

If the current repo name feels too narrow, a clearer title would be one of:

- `sustainable-procurement-nlp-pipeline`
- `public-procurement-sustainability-classifier`
- `sustainable-public-procurement-nlp`

Those names make the business domain and NLP scope clearer than a generic "supply chain classifier."
