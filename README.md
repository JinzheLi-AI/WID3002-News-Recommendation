# WID3002 Personalized News Recommendation System

Group Assignment G28 - WID3002 Natural Language Processing

## Overview

This project implements a personalized news recommendation system using the MINDsmall dataset from Microsoft News Dataset. The system ranks candidate news articles for each user by combining article text similarity, user reading behavior, collaborative filtering, popularity, category preference, and subcategory preference.

The final model is a hybrid ranking model. It is evaluated against single-strategy models and a popularity-based baseline using ranking metrics commonly used in recommendation tasks.

Deployed Streamlit web app:

https://spt2m5tpvsmaqvvnj32ick.streamlit.app

GitHub repository:

https://github.com/JinzheLi-AI/WID3002-News-Recommendation

## Project Objective

The goal is not to classify whether one news article is good or bad. The goal is to reorder a candidate list of news articles for a specific user, so that articles the user is more likely to click appear closer to the top of the list.

This follows the MIND candidate ranking setting:

1. Each impression contains one user and a list of candidate news articles.
2. Some candidate articles are clicked and others are not clicked.
3. The model assigns a ranking score to every candidate article.
4. The candidate articles are sorted from highest score to lowest score.
5. The ranking quality is evaluated by checking whether clicked articles appear near the top.

## Dataset

The project uses the MINDsmall dataset. For the full notebook experiment, the following files are required locally:

```text
MINDsmall_train/
|-- news.tsv
|-- behaviors.tsv

MINDsmall_dev/
|-- news.tsv
|-- behaviors.tsv
```

Only `news.tsv` and `behaviors.tsv` are required for this implementation. Entity embedding and relation embedding files are not used.

The deployed web app uses a lightweight demo subset for online accessibility, while the full notebook evaluates the complete MINDsmall validation output.

## Repository Structure

```text
WID3002-News-Recommendation/
|-- app.py
|-- WID3002_G28_NewsRecommendation.ipynb
|-- README.md
|-- requirements.txt
|-- metrics_comparison.csv
|-- model_comparison_results.csv
|-- ablation_results.csv
|-- data/
|   |-- articles_sample.csv
|   |-- demo_articles.csv
|   |-- demo_ranked_recommendations.csv
|-- results/
|   |-- evaluation_plot.png
|   |-- metrics_comparison.csv
```

Large raw datasets, full ranked recommendation files, embedding matrices, and pickle model files are excluded from GitHub because they are too large for normal repository upload and are not required for the online demo.

## System Architecture

The system contains five main stages.

### 1. Data Loading and Preprocessing

The system reads MIND news metadata and user behavior logs. News metadata contains article ID, category, subcategory, title, and abstract. Behavior logs contain user ID, reading history, impression ID, candidate article IDs, and click labels.

The candidate impressions are expanded so that each row represents one candidate article under one impression.

### 2. NLP Text Representation

Each news article is represented using its title and abstract. The full notebook creates text representations with:

- TF-IDF vectors for keyword-level matching
- Sentence embeddings for semantic similarity

These article vectors are used to compare user interest profiles with candidate articles.

### 3. User Profiling

The system builds user profiles from user reading history. If a user has read articles in the past, the system summarizes the user's interests using the text and topic information from those articles.

This allows the model to estimate whether a new candidate article matches the user's previous reading behavior.

### 4. Hybrid Recommendation Model

The final model combines five recommendation signals:

| Signal | Purpose |
|---|---|
| Content similarity | Measures how similar the candidate article is to the user's reading interests |
| Collaborative filtering | Uses behavior patterns from similar users |
| Popularity | Gives a prior score to articles that are generally clicked more often |
| Category preference | Matches broad user interests such as news, sports, finance, or entertainment |
| Subcategory preference | Matches more specific interests such as football, markets, technology, or music news |

The final score is calculated as:

```text
final_score =
    0.45 * content_score
  + 0.10 * cf_score
  + 0.20 * popularity_score
  + 0.10 * category_score
  + 0.15 * subcategory_score
```

After every candidate article receives a score, the system sorts the candidate list by score in descending order.

### 5. Evaluation

The system is evaluated using ranking metrics:

| Metric | Meaning |
|---|---|
| Precision@K | How many top-K recommended articles were clicked |
| Recall@K | How many clicked articles were retrieved in the top-K list |
| NDCG@K | Whether clicked articles appear near the top of the ranked list |

NDCG@10 is used as the main metric because ranking position matters. A clicked article at rank 1 is better than the same clicked article at rank 10.

## Model Comparison

To address controlled comparison, the project compares four complete model strategies:

| Model | Description |
|---|---|
| Hybrid Model | Combines all five recommendation signals |
| Content-based Model | Uses only content similarity |
| Popularity-based Model | Uses only popularity |
| Collaborative Filtering Model | Uses only collaborative filtering |

Final model comparison:

| Model | NDCG@10 | Performance |
|---|---:|---:|
| Hybrid Model | 0.4359 | 43.59% |
| Content-based Model | 0.4136 | 41.36% |
| Popularity-based Model | 0.3460 | 34.60% |
| Collaborative Filtering Model | 0.3126 | 31.26% |

The Hybrid Model achieves the best NDCG@10, so it is kept as the final model.

## Controlled Ablation Study

The ablation study starts from the final Hybrid Model and removes only one signal at a time. This avoids changing multiple variables at once and makes the contribution of each signal easier to interpret.

Summary:

| Variant | NDCG@10 | Interpretation |
|---|---:|---|
| Final Hybrid Model | 0.4359 | Best overall model |
| Without content score | 0.4167 | Content similarity contributes clearly |
| Without category score | 0.4304 | Category preference provides smaller but useful signal |
| Without subcategory score | 0.4209 | Subcategory preference contributes clearly |
| Without CF score | 0.4360 | CF contribution is very small in this setting |
| Without popularity score | 0.4360 | Popularity contribution is very small in this setting |

This shows that content similarity and subcategory preference are the strongest contributors in the final system.

## Streamlit Web App

The web app is implemented in `app.py`.

Main pages:

| Page | Function |
|---|---|
| Home | Shows system overview, workflow, and headline metrics |
| Evaluation | Shows metric tables, model comparison, and ablation results |
| Recommendation Demo | Lets users select a user ID or impression ID and view ranked recommendations |
| Custom News Ranking | Lets users enter an external news article and see where it ranks |
| Article Explorer | Lets users browse article metadata |
| About | Explains the model design and evaluation logic |

## External News Input

The Custom News Ranking page allows a user to enter a new article with:

- title
- abstract
- category
- subcategory

The system inserts the external article into an existing candidate list as `CUSTOM_NEWS` and recalculates the ranking.

Because a newly typed external article has no historical clicks, it cannot have a real collaborative filtering score. Therefore:

```text
cf_score = 0
```

The user also cannot manually control popularity. Instead, the system estimates external article popularity automatically:

1. Use the average popularity of historical articles in the same subcategory.
2. If unavailable, use the average popularity of historical articles in the same category.
3. If unavailable, use global average popularity.
4. If no popularity data exists, set popularity to 0.

This prevents users from manually manipulating the final ranking.

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the web app:

```bash
streamlit run app.py
```

The local app can use either:

- the lightweight demo files in `data/`, or
- the full MINDsmall files if `MINDsmall_train/` and `MINDsmall_dev/` are placed beside `app.py`

## Running the Notebook

Open:

```text
WID3002_G28_NewsRecommendation.ipynb
```

Run the notebook cells from top to bottom. The notebook generates:

- cleaned article data
- user behavior expansion
- text representations
- recommendation scores
- model comparison results
- ablation study results
- evaluation metrics
- ranked recommendation outputs

## Key Results

Final hybrid model results:

| Metric | Hybrid Model | Popularity Baseline | Improvement |
|---|---:|---:|---:|
| P@5 | 0.1209 | 0.0906 | 0.0303 |
| R@5 | 0.4795 | 0.3782 | 0.1013 |
| NDCG@5 | 0.3825 | 0.2784 | 0.1041 |
| P@10 | 0.0849 | 0.0688 | 0.0161 |
| R@10 | 0.6466 | 0.5489 | 0.0977 |
| NDCG@10 | 0.4359 | 0.3375 | 0.0984 |

The Hybrid Model improves NDCG@10 by about 29.2% over the popularity baseline.

## Notes on Deployment

The deployed Streamlit version uses lightweight demo files because full MINDsmall outputs and embedding files are too large for normal GitHub upload and Streamlit Cloud deployment.

The full experiment remains available in the notebook and local output files.

## Authors

WID3002 Group Assignment G28

