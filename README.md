# WID3002 Personalized News Recommendation System

Group Assignment G28 - WID3002 Natural Language Processing

## Overview

This project builds a personalized news recommendation system using the MINDsmall dataset. The model ranks candidate news articles for each user by combining article content, user reading history, collaborative filtering, popularity, category preference, and subcategory preference.

The final system uses a hybrid ranking approach and compares its performance with a popularity-based baseline.

## Dataset

Required MINDsmall files:

```text
MINDsmall_train/
├── news.tsv
└── behaviors.tsv

MINDsmall_dev/
├── news.tsv
└── behaviors.tsv
```

Only `news.tsv` and `behaviors.tsv` are required. The entity and relation embedding files are not used in this project.

Large raw dataset files are not included in this repository.

## Project Structure

```text
README.md
demo.py
WID3002_G28_NewsRecommendation_Final_Sections.ipynb
data/
├── demo_articles.csv
├── demo_ranked_recommendations.csv
└── articles_sample.csv
results/
├── metrics_comparison.csv
└── evaluation_plot.png
```

## Methodology

The system uses five recommendation signals:

| Signal | Description |
|---|---|
| content_score | Semantic similarity between user profile and candidate article |
| cf_score | Collaborative filtering score from similar users |
| popularity_score | Article popularity based on training clicks |
| category_score | Broad topic preference from user history |
| subcategory_score | More specific topic preference from user history |

Final hybrid score:

```text
final_score =
    0.45 * content_score +
    0.10 * cf_score +
    0.20 * popularity_score +
    0.10 * category_score +
    0.15 * subcategory_score
```

These weights achieved the best NDCG@10 among the tested weight combinations.

## Results

| Metric | Hybrid Model | Popularity Baseline | Improvement |
|---|---:|---:|---:|
| P@5 | 0.1209 | 0.0906 | 0.0303 |
| R@5 | 0.4795 | 0.3782 | 0.1013 |
| NDCG@5 | 0.3825 | 0.2784 | 0.1041 |
| P@10 | 0.0849 | 0.0688 | 0.0161 |
| R@10 | 0.6466 | 0.5489 | 0.0977 |
| NDCG@10 | 0.4359 | 0.3375 | 0.0984 |

The hybrid model improves NDCG@10 by 0.0984 over the popularity baseline, which is approximately a 29.2% relative improvement.

## How to Run the Notebook

1. Download and extract `MINDsmall_train` and `MINDsmall_dev`.
2. Place `news.tsv` and `behaviors.tsv` in the dataset folder used in the notebook.
3. Open and run:

```text
WID3002_G28_NewsRecommendation_Final_Sections.ipynb
```

The notebook generates the full recommendation output, evaluation metrics, and evaluation plot.

## Interactive Demo

Run:

```bash
python demo.py
```

The demo uses small sample files included in `data/`, so it can run without the full MIND dataset.

Demo options:

1. Show final evaluation results
2. Show example IDs
3. Search recommendations by `impression_id`
4. Search recommendations by `user_id`
5. Show clicked articles ranked by the model
6. Explain score columns

Example ID formats:

```text
impression_id: 1
user_id: U80234
news_id: N42844
```

## Notes

The final weights were selected using a validation subset. This does not guarantee globally optimal weights, but it provides a practical setting for the hybrid model.

Validation click labels are used for evaluation and weight selection only. They are not used directly as ranking features.
