# Interview Guide

## 30-Second Version

I worked on an NLP pipeline for public procurement documents as part of a sustainable procurement analytics project. The goal was to classify each RFP or RFQ by how strong its sustainability requirements were, from simple mention to mandatory pass/fail requirements. I handled the workflow end to end: collecting PDFs, extracting and cleaning text, designing document-level labels plus evidence spans, testing TF-IDF and semantic features, and thinking carefully about explainability and data leakage.

## 90-Second Version

The hard part was not just choosing a model. First, I had to define the sample unit correctly: one full procurement document equals one training sample. Then I added evidence-span tracking so the output could be justified with the exact clause and page number. From there, the NLP work included PDF ingestion, text extraction, rule-based cleaning, feature engineering, multiclass classification, and comparing interpretable baselines like TF-IDF plus Logistic Regression with semantic methods such as BERT embeddings. The broader project also involved scraping from CanadaBuys, handling a real tender-document corpus, and preparing outputs for dashboard use. The project taught me that label quality, traceability, and evaluation design matter as much as model complexity.

## If Someone Asks "Did You Really Use NLP?"

Yes. I used NLP in a practical applied sense:

- text extraction from PDFs
- cleaning and normalization
- tokenization-related decisions
- TF-IDF feature construction
- domain lexicon feature engineering
- supervised text classification
- BERT or sentence-embedding experiments
- evidence retrieval for explainable outputs

You can also mention that the project sat inside a broader analytics workflow:

- public web data collection
- document processing
- supervised labeling
- model prototyping
- dashboard export

## If Someone Asks "Why Not Just Use BERT?"

Because the dataset was relatively small and the business problem needed explainability. A TF-IDF baseline is easier to debug and gives a strong benchmark. Semantic models are useful, but they should improve on a leakage-safe baseline rather than replace it by default.

## If Someone Asks "What Was The Biggest Technical Lesson?"

The biggest lesson was that a model is only as good as the sample definition and labeling policy. If the team has not clearly defined whether the model predicts at the document, page, or sentence level, the whole pipeline becomes inconsistent.

## Safe, Honest Positioning

Use these phrases:

- "I built a prototype and then restructured it into a more reproducible pipeline."
- "I experimented with both traditional NLP and semantic embeddings."
- "I focused on evidence traceability, not just prediction output."
- "I learned to prevent leakage by splitting at the full-document level."

Avoid these phrases unless you can prove them:

- "The model is production-ready."
- "BERT solved the problem."
- "The accuracy proves robustness."
