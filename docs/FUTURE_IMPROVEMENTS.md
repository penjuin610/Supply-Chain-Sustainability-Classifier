# Future Improvements

## Goal

This document is for future maintainers and for explaining what should happen next after the current prototype stage.

## Current Recommendation

Do **not** jump directly to fine-tuning BERT as the next step.

The recommended order is:

1. stabilize labels and data schemas
2. build a strong TF-IDF baseline
3. add domain lexicon features
4. test sentence embeddings
5. fine-tune BERT only if the data volume and evaluation setup justify it

## Why TF-IDF Should Come First

Use `TF-IDF + Logistic Regression` first because it is:

- easy to reproduce
- easy to explain in interviews and reviews
- safer for a small labeled dataset
- useful for detecting leakage or labeling issues
- a strong benchmark that later models must beat

Recommended baseline:

```text
basic-cleaned document text
-> TF-IDF (unigrams + bigrams)
-> Logistic Regression
```

## When To Add Embeddings

Sentence embeddings or document embeddings are the best next semantic step when:

- labels are stable
- document-level and evidence-level tasks are separated
- you want better context handling without full fine-tuning

Recommended semantic route:

```text
chunk / evidence text
-> sentence transformer embedding
-> lightweight classifier
-> document-level aggregation
```

This often gives a better tradeoff than immediately fine-tuning BERT on a small corpus.

## When Fine-Tuning BERT Is Worth It

Fine-tune BERT only if most of the following are true:

- you have a much larger labeled dataset
- labels are consistent and reviewed
- long-document chunking is well defined
- you have a proper held-out test design
- the TF-IDF and embedding baselines are already documented
- you can justify the added complexity

If those conditions are not met, fine-tuning BERT can make the repo look more advanced while actually becoming less reliable.

## Highest-Value Improvements

### 1. Annotation redesign

Split annotation into:

- `document_labels.csv`
- `evidence_spans.csv`

This will make the task much easier to explain and evaluate.

### 2. Page-linked evidence retrieval

For every predicted level, retain:

- page number
- sentence or clause
- evidence type
- confidence

This is one of the most valuable business-facing improvements.

### 3. Better PDF handling

Improve handling for:

- scanned PDFs
- OCR fallback
- table extraction
- encrypted files
- poor-quality amendment documents

### 4. Domain features

Add engineered signals for procurement language:

- `must`
- `shall`
- `mandatory`
- `pass/fail`
- `evaluation criteria`
- percentages and weights
- `ISO 14001`
- `FSC`
- `LEED`

These features should be used in context, not as naive keyword counts.

### 5. Evaluation maturity

Add:

- macro F1
- per-class metrics
- confusion matrix
- class distribution
- error analysis by cause

## Suggested Roadmap

### Phase 1

- migrate notebook logic into scripts
- preserve raw / extracted / cleaned / labeled forms
- define stable schemas

### Phase 2

- run TF-IDF baseline on real labeled data
- inspect feature importance
- perform leakage-safe evaluation

### Phase 3

- add sentence embeddings
- compare against TF-IDF baseline
- test evidence aggregation strategy

### Phase 4

- consider fine-tuned BERT for chunk-level classification
- compare performance honestly
- keep only if it materially improves held-out results

## Recommended Positioning For Future Maintainers

This repository should be treated as:

- a real applied NLP prototype
- a handoff-friendly project
- a baseline-first modeling project
- an explainability-sensitive procurement analytics system

Not as:

- a pure deep-learning demo
- a BERT-first benchmark
- a finished production system

