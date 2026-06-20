# Project Context

## Why This Project Existed

This project was created to support the Sustainable Public Procurement Platform (SPPP), an initiative connected to the George Weston Ltd. Centre for Sustainable Supply Chains at York University.

The business motivation was broader than simple text classification. The platform aimed to help public institutions:

- benchmark how strongly sustainability appears in procurement documents;
- compare procurement practices across institutions and jurisdictions;
- identify better examples of supplier-facing sustainability requirements;
- improve future RFP and RFQ drafting;
- make sustainable procurement more visible and measurable.

## What The Client Wanted

Based on the original Statement of Work and final report, the project aimed to build a prototype analytical platform that could:

1. gather procurement tendering data from public sources;
2. parse and organize unstructured procurement documents;
3. classify sustainability content and commitment levels;
4. support reporting and benchmarking through dashboard outputs;
5. generate practical insights for public-sector procurement teams.

The broader original SOW included several analysis goals:

- sustainability issue identification
- sustainability commitment level classification
- tender size analysis
- jurisdictional classification
- site coverage evaluation

Over time, the practical scope narrowed to a more achievable prototype centered mainly on CanadaBuys.

## What The Team Actually Did

From the historical deliverables, the implemented workflow included:

- web scraping using Octoparse and Selenium
- PDF download automation from scraped links
- text extraction from tender documents
- manual sentence-level sustainability labeling
- five-level sustainability scale design
- BERT-based feature extraction experiments
- supervised classification prototype
- CSV output generation for reporting
- Tableau dashboard handoff

## Concrete Historical Numbers

The final report states:

- 5,657 total scraped documents
- 3,087 English documents
- 2,546 French documents
- 50 RFP documents manually reviewed in the early labeled subset
- 210 English sustainability-related sentences identified
- 63 French sustainability-related sentences identified

These numbers are useful in interviews because they show that the work was grounded in a real document collection and labeling effort rather than a toy demo.

## Important Truth About The Prototype

The prototype mixed two levels of analysis:

- document-level ambition for platform reporting
- sentence-level labeling for early training

That is one reason the project later became hard to explain cleanly. The model story becomes much clearer when we state explicitly:

```text
Document-level label = business output
Evidence-span or sentence-level label = explanation and training support
```

## What The Final Deliverables Show

The final deliverables show that the project was not just "a model notebook." It included:

- a client report
- a presentation
- a handover manual
- Excel-based outputs
- a Tableau workbook
- taxonomy references
- scraping process documentation

That means the strongest way to present this project is as an applied analytics prototype with NLP components, not merely as an isolated classifier.

## Best Interview Framing

The clearest framing is:

I worked on an applied NLP and analytics prototype for sustainable public procurement. We collected tender documents from CanadaBuys, extracted and cleaned PDF text, manually labeled sustainability evidence, experimented with NLP classification methods including BERT-based features, and produced dashboard-oriented outputs that could support procurement benchmarking and future policy analysis.

## Future Improvement Directions

The most credible future improvements are:

- move from notebook experiments to reproducible pipelines
- separate document labels from evidence labels
- add page-linked evidence extraction
- start with TF-IDF + Logistic Regression as a traceable baseline
- use sentence embeddings or chunk aggregation for semantic improvement
- improve OCR/table handling
- support cross-jurisdiction comparison
- refine taxonomy beyond a single scalar level
- connect outputs directly to dashboard drill-down
