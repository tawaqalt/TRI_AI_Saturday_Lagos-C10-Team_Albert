# TRI_AI_Saturday_Lagos-C10-Team_Albert
Optimising RAG Document Retrieval for Agronomic Advice
# 🌾 Optimising RAG Document Retrieval for Agronomic Advice

## 🌍 Project Overview
This project builds the retrieval engine at the heart of a Retrieval-Augmented Generation (RAG) system, designed to help smallholder farmers across Sub-Saharan Africa access accurate agronomic advice quickly.

Given a farmer's question (for example, "why are my maize leaves turning yellow?"), the system searches a knowledge base of 695 agricultural factsheets and returns the 5 most relevant documents, ranked from most to least relevant. Rather than generating advice directly, the project focuses on solving the retrieval bottleneck that any downstream RAG system depends on: if the right document never surfaces, no amount of AI-generated text afterward can fix that.

The dataset spans 21 countries across Sub-Saharan Africa, with Nigeria as the most represented country (133 of 695 documents), which shapes our team's particular focus and analysis throughout this project.

## 🎯 Objectives
- Explore and understand the provided Kaggle dataset (documents, training queries, relevance judgements, test queries).
- Build and evaluate a baseline retrieval method (TF-IDF).
- Improve on the baseline using BM25 (refined keyword matching).
- Implement dense retrieval using pretrained sentence transformer embeddings to capture semantic meaning beyond exact word overlap.
- Combine sparse and dense retrieval into a hybrid approach, optionally with cross-encoder re-ranking for higher precision.
- Evaluate all methods using nDCG@5 against the provided expert relevance judgements.
- Generate and submit a final ranked top-5 prediction per test query in the format required by the competition.

## ⚙️ Tools & Libraries
- Python
- Pandas, NumPy
- scikit-learn (TF-IDF, cosine similarity)
- rank_bm25 (BM25 retrieval)
- sentence-transformers (dense embeddings, cross-encoder re-ranking)
- Kaggle Notebooks (required development and submission environment)

## 📊 Dataset
Source: Provided through the Kaggle competition *Agricultural Extension RAG: Smart Retrieval for Farmers*. No additional data was collected or scraped; the project uses the supplied dataset as-is.

The dataset consists of four files:
- **documents.csv** — 695 agricultural factsheets (9 fields: document_id, title, text, source, crop, country, origin, source_url, license), covering crop diseases, pests, nutrient deficiencies, soil management, climate adaptation, and fertiliser advice.
- **train_queries.csv** — 308 training queries.
- **qrels_train.csv** — 4,194 query-document relevance judgements, graded 0 (not relevant) to 3 (highly relevant).
- **test_queries.csv** — 200 held-out test queries used for final evaluation.

Document origin is split between "synthetic" (637 documents) and "llm_grounded" (58 documents). Full details on dataset composition, licensing, country distribution, and known limitations are documented in `docs/data_card.pdf`.

## 🧩 Approach Highlights
- **Sparse retrieval (TF-IDF, BM25):** keyword-based matching, improved over the baseline using BM25's fairer handling of document length and term frequency.
- **Dense retrieval (sentence embeddings):** captures semantic similarity so that differently worded but conceptually related queries and documents can still be matched (e.g. "yellow leaves" ↔ "nitrogen deficiency").
- **Hybrid retrieval:** combines sparse and dense scores (via weighted fusion or reciprocal rank fusion) to catch what either method alone might miss, particularly the hard/near-miss documents deliberately included in the dataset to test true semantic understanding.

## 🤖 Evaluation
All methods are evaluated using **nDCG@5**, comparing each method's top-5 ranked documents per query against the graded relevance judgements provided in `qrels_train.csv`. This metric rewards not just retrieving a relevant document, but ranking the most relevant ones highest.

## 🧭 Repository Structure
```
C10-team-marula/
├── README.md
├── docs/
│   ├── problem_statement.pdf
│   ├── data_card.pdf
│   ├── impact_statement_card.pdf
│   ├── stakeholder_engagement.pdf
│   
├── scripts/
│   └── [all project scripts/code]
└── data/
    └── [project data]
```

## 🧠 Insights & Expected Outcomes
- A retrieval system that measurably outperforms simple keyword matching (TF-IDF) on nDCG@5.
- Clearer understanding of where retrieval struggles, particularly on informally phrased queries and documents from underrepresented countries in the dataset.
- A transparent, documented foundation for future work exploring how such a system could eventually support real farmers and extension workers, contingent on further expert review and real-world testing.

## 👥 Contributors
**Team:** Albert
**Team Lead:** [Tawakalitu Abdullahi Yusuf]
**Team Members:**
- Inameti Mercy


**Program:** TRI AI Saturday's AI/ML Cohot 10

## 📜 Acknowledgment
This project was developed as part of TRI AI Saturday's AI/ML cohort challenge, focusing on applying retrieval-augmented generation techniques to a real-world, high-impact agricultural problem.

## 🔗 References
- Agricultural Extension RAG: Smart Retrieval for Farmers — Kaggle Competition
- TRI AI Saturday's AI/ML
- Sentence Transformers
