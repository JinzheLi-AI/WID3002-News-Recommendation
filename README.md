# WID3002 Personalized News Recommendation System
**Group Assignment G28** — WID3002 Natural Language Processing

## Overview
A hybrid news recommendation system that combines content-based filtering and collaborative filtering with an NLP pipeline to deliver personalized article recommendations. Built on the [MIND dataset](https://msnews.github.io) (Microsoft News Dataset).

## System Architecture
The pipeline consists of four stages:

1. **Data Preprocessing** — Parse and clean the MIND dataset, producing `articles.csv` and interaction logs
2. **NLP Pipeline** — Text cleaning, tokenization (spaCy), TF-IDF vectorization, and sentence embeddings (MiniLM-L6-v2)
3. **User Profiling** — Build recency-weighted user interest vectors; popularity-based cold-start fallback for new users
4. **Recommendation Engine** — Hybrid scoring: `final_score = α × content_score + (1-α) × collab_score`

## Dataset
[MIND-small](https://msnews.github.io) — 61,823 news articles, 8.5M user interaction records across 50,000 users.

> Download the dataset separately and place `MINDsmall_train/` and `MINDsmall_dev/` in the project root before running.

## Results

| Metric | Hybrid Model | Popularity Baseline |
|--------|-------------|-------------------|
| P@5 | 0.0011 | 0.0000 |
| R@5 | 0.0024 | 0.0000 |
| NDCG@5 | 0.0020 | 0.0000 |
| P@10 | 0.0006 | 0.0001 |
| R@10 | 0.0025 | 0.0004 |
| NDCG@10 | 0.0020 | 0.0002 |

## Repository Structure
```
├── WID3002_G28_NewsRecommendation.ipynb   # Main notebook (all 5 stages)
├── data/
│   ├── articles.csv                        # Cleaned news articles
│   ├── article_id_index.json               # Article ID to index mapping
│   ├── popular_by_category.json            # Cold-start fallback
│   └── ranked_recommendations.csv          # Top-10 recommendations per user
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.pkl
│   ├── article_embeddings.npy              # MiniLM-L6-v2 embeddings (384-dim)
│   └── user_profiles.pkl                   # User interest vectors
└── results/
    ├── metrics_comparison.csv
    └── evaluation_plots.png
```

## Requirements
```
pandas
numpy
spacy
scikit-learn
sentence-transformers
scipy
matplotlib
```
```
python -m spacy download en_core_web_sm
```
