# TRI_AI_Saturday_Lagos-C10-Team_Albert

# 🌾 Optimising RAG Document Retrieval for Agronomic Advice

## 1. Dataset

The dataset was provided through the Kaggle competition *Agricultural Extension RAG: Smart Retrieval for Farmers*. No additional data was collected or scraped; the project uses the supplied dataset as-is, selected because it was the fixed, self-contained corpus required by the competition.

It consists of four files:
- **documents.csv** — 695 agricultural factsheets (9 fields: document_id, title, text, source, crop, country, origin, source_url, license), covering crop diseases, pests, nutrient deficiencies, soil management, climate adaptation, and fertiliser advice.
- **train_queries.csv** — 308 training queries.
- **qrels_train.csv** — 4,194 query-document relevance judgements, graded 0 (not relevant) to 3 (highly relevant).
- **test_queries.csv** — 200 held-out test queries used for final evaluation.

The corpus spans 21 countries across Sub-Saharan Africa, with Nigeria as the most represented country (133 of 695 documents), followed by Ghana (103), Tanzania (92), and Kenya (75). Document origin is split between "synthetic" (637 documents) and "llm_grounded" (58 documents), and licensing follows the same split, mostly "synthetic (CC0)" with a small number of real open-licensed documents (CC-BY-4.0, CC-BY-SA-4.0, CC0-1.0, CC-BY-3.0). Full details on dataset composition, bias, and limitations are documented in `docs/data_card.pdf`.

## 2. Training Pipeline

This is a retrieval and ranking task, not a trained classifier, so there is no model training in the traditional sense. Our pipeline instead consists of:

- **Preprocessing:** document title and text fields are concatenated into a single text block per document (`title + ". " + text`) to preserve compact topic signal from the title alongside full content.
- **Sparse retrieval:** a TF-IDF baseline, improved using BM25 (`rank_bm25`), tuned with `k1=1.2, b=0.75`.
- **Dense retrieval:** pretrained sentence embeddings (`BAAI/bge-large-en-v1.5`), using the model's recommended query instruction prefix for retrieval accuracy.
- **Hybrid fusion:** BM25 and dense similarity scores are min-max normalised and combined with a weighted sum (`alpha = 0.6` favouring the dense score), narrowing the corpus to a top-30 candidate pool per query.
- **Re-ranking:** the top-30 candidates are passed through a cross-encoder (`BAAI/bge-reranker-large`), which jointly scores each (query, document) pair for higher precision, producing the final top-5 ranking.

Key design choices, such as the hybrid weighting and candidate pool size, were selected based on standard practice for two-stage retrieval pipelines and refined using local evaluation (see below) rather than formal hyperparameter search, given project time constraints.

## 3. Evaluation

All methods are evaluated using **nDCG@5**, comparing each method's top-5 ranked documents per query against the graded relevance judgements in `qrels_train.csv`. This metric rewards not just retrieving a relevant document, but ranking the most relevant ones highest.

We verify our method locally before submitting: `train_queries.csv` and `qrels_train.csv` are used to compute nDCG@5 on training data ahead of generating predictions on the held-out `test_queries.csv`, so performance can be checked without relying solely on competition submission feedback. We specifically reviewed queries involving near-miss documents (topically similar but incorrect matches) to check whether hybrid and re-ranked retrieval reduced this failure mode compared to the TF-IDF baseline.

## 4. Reproduction

To reproduce our results from scratch, run the scripts in `scripts/` in the following order:

1. `data_loading.py` — loads and validates all four dataset files.
2. `bm25_retrieval.py` — builds the BM25 sparse index.
3. `dense_retrieval.py` — loads the dense embedding model and encodes all documents.
4. `hybrid_retrieval.py` — combines BM25 and dense scores, then applies cross-encoder re-ranking.
5. `metrics.py` — computes nDCG@5 on training data to verify performance before submission.
6. `run_hybrid_submission.py` — runs the full pipeline end-to-end and generates `submission.csv` in the format required by the competition.

Required packages are listed in `requirements.txt`. This project was developed and must be run within a Kaggle Notebook environment, with the competition dataset attached via the Input panel.

## 5. Appendix — Contributors & Mentors

**Team:** Albert

**Team Lead:** Tawakalitu Abdullahi Yusuf

**Team Members:**
- Inameti Mercy
- Ibrahim Keji

**Mentors:**
- Seun Ajayi
- Adnan Haddy
- Samuel Taiwo

**Program:** TRI AI Saturday's AI/ML Cohort 10

## 🧭 Repository Structure
```
C10-Team-Albert/
├── README.md
├── docs/
│   ├── problem_statement.pdf
│   ├── data_card.pdf
│   ├── impact_statement_card.pdf
│   └── stakeholder_engagement.pdf
├── scripts/
│   └── [all project scripts/code]
└── data/
    └── [project data]
```

## 📜 Acknowledgment
This project was developed as part of TRI AI Saturday's AI/ML cohort challenge, focusing on applying retrieval-augmented generation techniques to a real-world, high-impact agricultural problem.

## 🔗 References
- Agricultural Extension RAG: Smart Retrieval for Farmers — Kaggle Competition
- TRI AI Saturday's AI/ML
