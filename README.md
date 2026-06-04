# WID3002 Personalized News Recommendation System

Group Assignment G28 - WID3002 Natural Language Processing

## Overview

This project implements a personalized news recommendation system using the MINDsmall dataset. The system ranks candidate news articles for each user by combining article content, user reading history, collaborative filtering, popularity, category preference, and subcategory preference.

The final model follows a hybrid ranking approach. Its performance is compared with a popularity-based baseline using ranking metrics commonly used in recommendation tasks.

## Dataset

The project uses the MINDsmall dataset from Microsoft News Dataset.

Required files:

```text
MINDsmall_train/
|-- news.tsv
`-- behaviors.tsv

MINDsmall_dev/
|-- news.tsv
`-- behaviors.tsv
```

Only `news.tsv` and `behaviors.tsv` are required. The entity and relation embedding files are not used in this implementation.

Large raw dataset files are not included in this repository.

## Project Structure

```text
README.md
requirements.txt
app.py
demo.py
WID3002_G28_NewsRecommendation.ipynb

data/
|-- demo_articles.csv
|-- demo_ranked_recommendations.csv
`-- articles_sample.csv

results/
|-- metrics_comparison.csv
`-- evaluation_plot.png
```

## Methodology

The recommendation engine uses five ranking signals:

| Signal | Description |
|---|---|
| `content_score` | Measures similarity between the user's reading profile and each candidate article |
| `cf_score` | Uses collaborative filtering patterns from users with similar reading behavior |
| `popularity_score` | Gives higher scores to articles with stronger click popularity in the training data |
| `category_score` | Matches candidate articles with the user's preferred news categories |
| `subcategory_score` | Adds a more specific topic preference based on article subcategories |

Final hybrid score:

```text
final_score =
    0.45 * content_score +
    0.10 * cf_score +
    0.20 * popularity_score +
    0.10 * category_score +
    0.15 * subcategory_score
```

These weights were selected because they produced the best NDCG@10 among the tested weight combinations.

## Evaluation

This is a recommendation ranking task, so the model is evaluated with ranking metrics instead of classification accuracy.

| Metric | Purpose |
|---|---|
| `P@K` | Measures how many recommended articles in the top K are actually clicked |
| `R@K` | Measures how many clicked articles are successfully retrieved within the top K |
| `NDCG@K` | Measures whether clicked articles are ranked near the top of the recommendation list |

`P@10` means Precision@10 and `R@10` means Recall@10. These match the evaluation direction stated in the proposal.

## Results

| Metric | Hybrid Model | Popularity Baseline | Improvement |
|---|---:|---:|---:|
| P@5 | 0.1209 | 0.0906 | 0.0303 |
| R@5 | 0.4795 | 0.3782 | 0.1013 |
| NDCG@5 | 0.3825 | 0.2784 | 0.1041 |
| P@10 | 0.0849 | 0.0688 | 0.0161 |
| R@10 | 0.6466 | 0.5489 | 0.0977 |
| NDCG@10 | 0.4359 | 0.3375 | 0.0984 |

The hybrid model improves NDCG@10 by 0.0984 over the popularity baseline, which is about a 29.2% relative improvement.

## Web Demo

The project includes an interactive Streamlit web application:

```text
app.py
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the web demo:

```bash
streamlit run app.py
```

If the `streamlit` command is not available, run:

```bash
python -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

The web demo includes:

| Page | Function |
|---|---|
| Home | Shows the system overview, workflow, data status, and main performance result |
| Evaluation | Displays the evaluation table, baseline comparison, and explanation of ranking metrics |
| Recommendation Demo | Lets users search recommendations by `user_id` or `impression_id` |
| Article Explorer | Allows browsing and filtering news articles by category or keyword |
| About | Summarizes the model design, evaluation method, and project limitations |

The recommendation page shows ranked articles, clicked labels, final scores, score breakdowns, and article details.

Example ID formats:

```text
user_id: U38418
impression_id: 8
news_id: N20477
```

## Terminal Demo

A command-line demo is also included:

```bash
python demo.py
```

The terminal demo can show evaluation results, example IDs, recommendation lists, clicked articles, and score explanations.

## How to Run the Notebook

1. Download and extract `MINDsmall_train` and `MINDsmall_dev`.
2. Place `news.tsv` and `behaviors.tsv` in the dataset folder.
3. Open and run:

```text
WID3002_G28_NewsRecommendation.ipynb
```

The notebook generates the recommendation output, evaluation metrics, and evaluation plot.

## Notes

The final weights were selected using validation data. This gives a practical and measurable improvement over the baseline, but it does not prove a mathematically global optimum over all possible weight combinations.

Validation click labels are used for evaluation and weight tuning only. They are not used directly as ranking features.
