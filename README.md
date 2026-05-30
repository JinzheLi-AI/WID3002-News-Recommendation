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

## Results

| Metric | Hybrid Model | Popularity Baseline |
|--------|-------------|-------------------|
| P@5 | 0.0011 | 0.0000 |
| R@5 | 0.0024 | 0.0000 |
| NDCG@5 | 0.0020 | 0.0000 |
| P@10 | 0.0006 | 0.0001 |
| R@10 | 0.0025 | 0.0004 |
| NDCG@10 | 0.0020 | 0.0002 |

## Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/JinzheLi-AI/WID3002-News-Recommendation.git
cd WID3002-News-Recommendation
```

### 2. Download the MIND-small dataset
Go to https://msnews.github.io, agree to the license terms, and download:
- **MIND-small Training Set**
- **MIND-small Validation Set**

Extract both zip files into the project folder so the structure looks like:
```
WID3002-News-Recommendation/
├── MINDsmall_train/
│   ├── news.tsv
│   └── behaviors.tsv
├── MINDsmall_dev/
│   ├── news.tsv
│   └── behaviors.tsv
└── ...
```

### 3. Install dependencies
```bash
pip install pandas numpy spacy scikit-learn sentence-transformers scipy matplotlib
python -m spacy download en_core_web_sm
```

### 4. Run the notebook
Open `WID3002_G28_NewsRecommendation.ipynb` in VS Code or Jupyter and click **Run All**.

This will take ~20 minutes (mostly the sentence embedding step). Once complete, all model files will be generated locally.

### 5. Get recommendations
```bash
python demo.py
```
Enter any user ID (e.g. `U100`, `U500`, `U1000`) to see personalized recommendations.

## Repository Structure
```
├── WID3002_G28_NewsRecommendation.ipynb   # Main notebook (all stages)
├── demo.py                                 # Quick inference script
├── data/
│   ├── articles.csv
│   ├── article_id_index.json
│   ├── popular_by_category.json
│   └── ranked_recommendations.csv
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.pkl
│   ├── article_embeddings.npy
│   └── user_profiles.pkl
└── results/
    ├── metrics_comparison.csv
    └── evaluation_plots.png
```

> **Note:** `MINDsmall_train/` and `MINDsmall_dev/` are not included in this repo. Download them separately from https://msnews.github.io.
