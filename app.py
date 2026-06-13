import re
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

ARTICLE_FILES = [
    BASE_DIR / "articles.csv",
    DATA_DIR / "demo_articles.csv",
    DATA_DIR / "articles_sample.csv",
    DATA_DIR / "articles.csv",
]

NEWS_COLS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

NEWS_TSV_FILES = [
    BASE_DIR / "MINDsmall_train" / "news.tsv",
    BASE_DIR / "MINDsmall_dev" / "news.tsv",
]

RECOMMENDATION_FILES = [
    BASE_DIR / "ranked_recommendations.csv",
    DATA_DIR / "demo_ranked_recommendations.csv",
    DATA_DIR / "ranked_recommendations.csv",
]

METRICS_FILES = [
    BASE_DIR / "metrics_comparison.csv",
    RESULTS_DIR / "metrics_comparison.csv",
]

MODEL_COMPARISON_FILES = [
    BASE_DIR / "model_comparison_results.csv",
    DATA_DIR / "model_comparison_results.csv",
    RESULTS_DIR / "model_comparison_results.csv",
]

ABLATION_FILES = [
    BASE_DIR / "ablation_results.csv",
    DATA_DIR / "ablation_results.csv",
    RESULTS_DIR / "ablation_results.csv",
]

MODEL_SCORE_COLUMNS = {
    "Hybrid Model": "score",
    "Popularity-based Model": "popularity_score",
    "Content-based Model": "content_score",
    "Collaborative Filtering Model": "cf_score",
}

MODEL_DESCRIPTIONS = {
    "Hybrid Model": (
        "Final model. Combines content similarity, collaborative filtering, popularity, "
        "category preference, and subcategory preference."
    ),
    "Popularity-based Model": (
        "Ranks articles only by historical click popularity. It is stable but not personalized."
    ),
    "Content-based Model": (
        "Ranks articles using only content similarity between the user profile and article text."
    ),
    "Collaborative Filtering Model": (
        "Ranks articles using similar users' click behavior. Missing interaction history gives a score of 0."
    ),
}

FINAL_WEIGHTS = {
    "content": 0.45,
    "cf": 0.10,
    "popularity": 0.20,
    "category": 0.10,
    "subcategory": 0.15,
}

SIGNAL_COLUMNS = {
    "content": "content_score",
    "cf": "cf_score",
    "popularity": "popularity_score",
    "category": "category_score",
    "subcategory": "subcategory_score",
}

EXTERNAL_NEWS_EXAMPLES = {
    "AI study tool": {
        "title": "New AI tool helps students summarize daily news faster",
        "abstract": (
            "A newly released artificial intelligence tool is designed to help students "
            "read, summarize, and compare news articles more efficiently."
        ),
        "category": "news",
        "subcategory": "technology",
    },
    "College football": {
        "title": "College football team prepares for championship after strong season",
        "abstract": (
            "The team is entering the championship game after a successful season with "
            "strong defensive performances and several close wins."
        ),
        "category": "sports",
        "subcategory": "football_ncaa",
    },
    "Market update": {
        "title": "Stock market rises as investors react to lower inflation data",
        "abstract": (
            "Major stock indexes moved higher after new inflation data suggested that "
            "price growth may be slowing."
        ),
        "category": "finance",
        "subcategory": "markets",
    },
    "Music tour": {
        "title": "Popular singer announces new world tour after album release",
        "abstract": (
            "The artist announced a new international tour following the release of a "
            "successful album, and tickets are expected to sell quickly."
        ),
        "category": "music",
        "subcategory": "musicnews",
    },
}


st.set_page_config(
    page_title="News Recommendation System",
    page_icon="📰",
    layout="wide",
)


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_articles():
    path = first_existing(ARTICLE_FILES)
    if path is not None:
        return pd.read_csv(path), path

    frames = []
    for news_path in NEWS_TSV_FILES:
        if news_path.exists():
            frame = pd.read_csv(
                news_path,
                sep="\t",
                header=None,
                names=NEWS_COLS,
                usecols=["news_id", "category", "subcategory", "title", "abstract"],
                dtype=str,
            )
            frames.append(frame)

    if not frames:
        return pd.DataFrame(), None

    articles = pd.concat(frames, ignore_index=True)
    articles = articles.drop_duplicates(subset="news_id").reset_index(drop=True)
    return articles, "MINDsmall_train/news.tsv + MINDsmall_dev/news.tsv"


@st.cache_data(show_spinner=False)
def load_recommendations():
    path = first_existing(RECOMMENDATION_FILES)
    if path is None:
        return pd.DataFrame(), None
    return pd.read_csv(path), path


@st.cache_data(show_spinner=False)
def load_metrics():
    path = first_existing(METRICS_FILES)
    if path is None:
        return pd.DataFrame(), None
    return pd.read_csv(path), path


@st.cache_data(show_spinner=False)
def load_model_comparison():
    path = first_existing(MODEL_COMPARISON_FILES)
    if path is None:
        return pd.DataFrame(), None
    return pd.read_csv(path), path


@st.cache_data(show_spinner=False)
def load_ablation_results():
    path = first_existing(ABLATION_FILES)
    if path is None:
        return pd.DataFrame(), None
    return pd.read_csv(path), path


def log2(value):
    import math

    return math.log(value, 2)


def label_array(df):
    if "label" not in df.columns:
        return pd.Series([0] * len(df), index=df.index)
    return pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)


def precision_at_k(labels, k):
    labels = list(labels)[:k]
    if not labels:
        return 0.0
    return float(sum(labels) / len(labels))


def recall_at_k(labels, k):
    labels = list(labels)
    positives = sum(labels)
    if positives == 0:
        return 0.0
    return float(sum(labels[:k]) / positives)


def ndcg_at_k(labels, k):
    labels = list(labels)[:k]
    if not labels:
        return 0.0
    dcg = sum(label / log2(rank + 2) for rank, label in enumerate(labels))
    ideal = sorted(labels, reverse=True)
    idcg = sum(label / log2(rank + 2) for rank, label in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def weighted_signal_score(df, weights):
    score = pd.Series([0.0] * len(df), index=df.index, dtype=float)
    for signal, weight in weights.items():
        column = SIGNAL_COLUMNS[signal]
        if column in df.columns:
            score += weight * pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return score


def model_score_series(df, model_name):
    if model_name == "Hybrid Model" and "score" not in df.columns:
        return weighted_signal_score(df, FINAL_WEIGHTS)

    column = MODEL_SCORE_COLUMNS.get(model_name, "score")
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(0.0)
    return pd.Series([0.0] * len(df), index=df.index)


def rank_for_model(df, model_name):
    ranked = df.copy()
    ranked["score"] = model_score_series(ranked, model_name)
    ranked["model_name"] = model_name
    ranked = ranked.sort_values(["score", "news_id"], ascending=[False, True]).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked


def evaluate_score(recs, score_values):
    empty_result = {
        "precision_at_5": 0.0,
        "recall_at_5": 0.0,
        "ndcg_at_5": 0.0,
        "precision_at_10": 0.0,
        "recall_at_10": 0.0,
        "ndcg_at_10": 0.0,
    }
    if recs.empty or "impression_id" not in recs.columns or "label" not in recs.columns:
        return empty_result

    temp = recs[["impression_id", "news_id", "label"]].copy()
    temp["score"] = pd.Series(score_values, index=recs.index).astype(float)
    scores = {key: [] for key in ["p5", "r5", "n5", "p10", "r10", "n10"]}

    for _, group in temp.groupby("impression_id", sort=False):
        labels = label_array(group)
        if len(group) < 2 or labels.sum() == 0:
            continue
        ranked = group.assign(label=labels).sort_values(["score", "news_id"], ascending=[False, True])
        ranked_labels = ranked["label"].tolist()
        scores["p5"].append(precision_at_k(ranked_labels, 5))
        scores["r5"].append(recall_at_k(ranked_labels, 5))
        scores["n5"].append(ndcg_at_k(ranked_labels, 5))
        scores["p10"].append(precision_at_k(ranked_labels, 10))
        scores["r10"].append(recall_at_k(ranked_labels, 10))
        scores["n10"].append(ndcg_at_k(ranked_labels, 10))

    def mean(values):
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "precision_at_5": mean(scores["p5"]),
        "recall_at_5": mean(scores["r5"]),
        "ndcg_at_5": mean(scores["n5"]),
        "precision_at_10": mean(scores["p10"]),
        "recall_at_10": mean(scores["r10"]),
        "ndcg_at_10": mean(scores["n10"]),
    }


def build_model_comparison(recs):
    rows = []
    for model_name in MODEL_SCORE_COLUMNS:
        metrics = evaluate_score(recs, model_score_series(recs, model_name))
        rows.append(
            {
                "model_name": model_name,
                **metrics,
                "performance_percentage": metrics["ndcg_at_10"] * 100,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("ndcg_at_10", ascending=False).reset_index(drop=True)
    return result


def renormalized_weights_without(signal_to_remove):
    weights = {key: value for key, value in FINAL_WEIGHTS.items() if key != signal_to_remove}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()} if total else weights


def build_ablation_results(recs):
    variants = [("Final Hybrid Model", "None", FINAL_WEIGHTS)]
    for signal in FINAL_WEIGHTS:
        variants.append(
            (
                f"Without {signal.replace('_', ' ').title()}",
                f"{signal}_score",
                renormalized_weights_without(signal),
            )
        )

    rows = []
    final_ndcg = None
    for variant_name, removed_signal, weights in variants:
        metrics = evaluate_score(recs, weighted_signal_score(recs, weights))
        if variant_name == "Final Hybrid Model":
            final_ndcg = metrics["ndcg_at_10"]
        rows.append(
            {
                "variant_name": variant_name,
                "removed_signal": removed_signal,
                **metrics,
                "performance_percentage": metrics["ndcg_at_10"] * 100,
            }
        )

    result = pd.DataFrame(rows)
    if final_ndcg is not None and not result.empty:
        result["drop_from_final_hybrid"] = final_ndcg - result["ndcg_at_10"]
    return result


def prepare_data():
    articles, articles_path = load_articles()
    recs, recs_path = load_recommendations()
    metrics, metrics_path = load_metrics()

    if not recs.empty and not articles.empty and "title" not in recs.columns:
        article_cols = [
            col
            for col in ["news_id", "category", "subcategory", "title", "abstract"]
            if col in articles.columns
        ]
        recs = recs.merge(articles[article_cols], on="news_id", how="left")

    return articles, recs, metrics, articles_path, recs_path, metrics_path


def metric_value(metrics, metric, column):
    if metrics.empty:
        return None
    row = metrics[metrics["Metric"] == metric]
    if row.empty or column not in metrics.columns:
        return None
    return float(row[column].iloc[0])


def section_header(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def score_label(value):
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.4f}"


def text_tokens(text):
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def combined_text(row):
    title = row.get("title", "")
    abstract = row.get("abstract", "")
    return f"{title} {abstract}"


def overlap_score(source_texts, candidate_text):
    source_tokens = set()
    for text in source_texts:
        source_tokens.update(text_tokens(text))

    candidate_tokens = text_tokens(candidate_text)
    if not source_tokens or not candidate_tokens:
        return 0.0

    return len(source_tokens & candidate_tokens) / len(candidate_tokens)


def preference_score(values, candidate_value):
    values = [str(value).lower() for value in values if pd.notna(value)]
    candidate_value = str(candidate_value).lower().strip()
    if not values or not candidate_value:
        return 0.0

    counts = pd.Series(values).value_counts(normalize=True)
    return float(counts.get(candidate_value, 0.0))


def estimate_external_popularity(reference_data, category, subcategory):
    if reference_data.empty or "popularity_score" not in reference_data.columns:
        return 0.0, "Popularity score is unavailable, so the value is set to 0."

    popularity = pd.to_numeric(reference_data["popularity_score"], errors="coerce")
    category_value = str(category).lower().strip()
    subcategory_value = str(subcategory).lower().strip()

    if "subcategory" in reference_data.columns and subcategory_value:
        subcategory_mask = reference_data["subcategory"].astype(str).str.lower().str.strip() == subcategory_value
        subcategory_scores = popularity[subcategory_mask].dropna()
        if not subcategory_scores.empty:
            value = float(subcategory_scores.mean())
            return min(max(value, 0.0), 1.0), "Estimated from historical articles in the same subcategory."

    if "category" in reference_data.columns and category_value:
        category_mask = reference_data["category"].astype(str).str.lower().str.strip() == category_value
        category_scores = popularity[category_mask].dropna()
        if not category_scores.empty:
            value = float(category_scores.mean())
            return min(max(value, 0.0), 1.0), "Estimated from historical articles in the same category."

    global_scores = popularity.dropna()
    if global_scores.empty:
        return 0.0, "Popularity score is unavailable, so the value is set to 0."

    value = float(global_scores.mean())
    return min(max(value, 0.0), 1.0), "Estimated from the global average popularity score."


def custom_news_scores(context, title, abstract, category, subcategory, reference_data):
    if "label" in context.columns:
        clicked = context[pd.to_numeric(context["label"], errors="coerce").fillna(0).astype(int) == 1]
    else:
        clicked = pd.DataFrame()
    history_like = clicked if not clicked.empty else context.sort_values("rank").head(5)
    source_texts = history_like.apply(combined_text, axis=1).tolist()
    candidate_text = f"{title} {abstract}"

    content = overlap_score(source_texts, candidate_text)
    category_pref = preference_score(history_like.get("category", pd.Series(dtype=str)), category)
    subcategory_pref = preference_score(history_like.get("subcategory", pd.Series(dtype=str)), subcategory)
    popularity_score, popularity_source = estimate_external_popularity(reference_data, category, subcategory)

    scores = {
        "content_score": min(content, 1.0),
        "cf_score": 0.0,
        "popularity_score": popularity_score,
        "category_score": category_pref,
        "subcategory_score": subcategory_pref,
    }
    scores["score"] = custom_score_for_model(scores, "Hybrid Model")
    return scores, popularity_source


def custom_score_for_model(scores, model_name):
    if model_name == "Popularity-based Model":
        return scores["popularity_score"]
    if model_name == "Content-based Model":
        return scores["content_score"]
    if model_name == "Collaborative Filtering Model":
        return scores["cf_score"]

    return (
        FINAL_WEIGHTS["content"] * scores["content_score"]
        + FINAL_WEIGHTS["cf"] * scores["cf_score"]
        + FINAL_WEIGHTS["popularity"] * scores["popularity_score"]
        + FINAL_WEIGHTS["category"] * scores["category_score"]
        + FINAL_WEIGHTS["subcategory"] * scores["subcategory_score"]
    )


def sidebar_controls(articles, recs, metrics_path, recs_path, articles_path):
    st.sidebar.title("News Recommender")
    st.sidebar.caption("Personalized candidate ranking demo")

    st.sidebar.markdown("### Data Status")
    st.sidebar.write(f"Articles: **{len(articles):,}**" if not articles.empty else "Articles: not found")
    st.sidebar.write(f"Recommendation rows: **{len(recs):,}**" if not recs.empty else "Recommendations: not found")
    st.sidebar.write(f"Metrics file: **{metrics_path.name}**" if metrics_path else "Metrics file: not found")

    with st.sidebar.expander("Loaded files"):
        st.write("Articles:", str(articles_path) if articles_path else "Not found")
        st.write("Recommendations:", str(recs_path) if recs_path else "Not found")
        st.write("Metrics:", str(metrics_path) if metrics_path else "Not found")

    st.sidebar.markdown("### How to Use")
    st.sidebar.write("1. Open **Recommendation Demo**.")
    st.sidebar.write("2. Choose a user or impression.")
    st.sidebar.write("3. Inspect ranked news and score breakdown.")


def home_page(articles, recs, metrics):
    st.title("Personalized News Recommendation System")
    st.write(
        "A hybrid recommendation system that ranks candidate news articles using content similarity, "
        "collaborative filtering, popularity, category preference, and subcategory preference."
    )

    ndcg10 = metric_value(metrics, "NDCG@10", "Hybrid Model")
    baseline = metric_value(metrics, "NDCG@10", "Popularity Baseline")
    recall10 = metric_value(metrics, "R@10", "Hybrid Model")
    precision10 = metric_value(metrics, "P@10", "Hybrid Model")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hybrid NDCG@10", score_label(ndcg10))
    c2.metric("Baseline NDCG@10", score_label(baseline))
    c3.metric("Recall@10", score_label(recall10))
    c4.metric("Precision@10", score_label(precision10))

    if ndcg10 is not None and baseline is not None:
        improvement = ndcg10 - baseline
        relative = improvement / baseline * 100 if baseline else 0
        st.success(
            f"The hybrid model improves NDCG@10 by {improvement:.4f}, "
            f"which is about {relative:.1f}% over the popularity baseline."
        )

    st.markdown("### System Workflow")
    workflow = pd.DataFrame(
        [
            ["1", "Data Preprocessing", "Clean news articles and expand user impression logs."],
            ["2", "NLP Representation", "Represent article text using TF-IDF and sentence embeddings."],
            ["3", "User Profiling", "Build interest profiles from user reading history."],
            ["4", "Hybrid Ranking", "Combine content, CF, popularity, category, and subcategory scores."],
            ["5", "Evaluation", "Compare against popularity baseline using P@K, R@K, and NDCG@K."],
        ],
        columns=["Step", "Module", "Purpose"],
    )
    st.dataframe(workflow, use_container_width=True, hide_index=True)

    st.markdown("### Dataset Snapshot")
    d1, d2, d3 = st.columns(3)
    d1.metric("Articles Loaded", f"{len(articles):,}" if not articles.empty else "N/A")
    d2.metric("Ranked Rows", f"{len(recs):,}" if not recs.empty else "N/A")
    if not recs.empty and "user_id" in recs.columns:
        d3.metric("Users in Demo Data", f"{recs['user_id'].nunique():,}")
    else:
        d3.metric("Users in Demo Data", "N/A")


def display_model_comparison_table(model_comparison):
    if model_comparison.empty:
        st.warning("No model comparison results available.")
        return

    table = model_comparison.copy().sort_values("ndcg_at_10", ascending=False).reset_index(drop=True)
    table.insert(0, "Rank", table.index + 1)
    table = table.rename(
        columns={
            "model_name": "Model",
            "precision_at_5": "Precision@5",
            "recall_at_5": "Recall@5",
            "ndcg_at_5": "NDCG@5",
            "precision_at_10": "Precision@10",
            "recall_at_10": "Recall@10",
            "ndcg_at_10": "NDCG@10",
            "performance_percentage": "Performance Percentage",
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)


def display_ablation_table(ablation_results):
    if ablation_results.empty:
        st.warning("No ablation results available.")
        return

    table = ablation_results.copy()
    table = table.rename(
        columns={
            "variant_name": "Variant",
            "removed_signal": "Removed Signal",
            "ndcg_at_10": "NDCG@10",
            "performance_percentage": "Performance Percentage",
            "drop_from_final_hybrid": "Drop from Final Hybrid",
        }
    )
    display_cols = [
        "Variant",
        "Removed Signal",
        "NDCG@10",
        "Performance Percentage",
        "Drop from Final Hybrid",
    ]
    display_cols = [col for col in display_cols if col in table.columns]
    st.dataframe(table[display_cols], use_container_width=True, hide_index=True)


def evaluation_page(metrics, model_comparison, ablation_results):
    section_header(
        "Evaluation Results",
        "Recommendation quality is evaluated using Precision@K, Recall@K, and NDCG@K.",
    )

    st.info(
        "This is a ranking task, so NDCG@10 is used as the main performance metric "
        "instead of normal accuracy. Performance Percentage = NDCG@10 × 100%."
    )

    st.markdown("### Model Performance Ranking")
    display_model_comparison_table(model_comparison)

    if not model_comparison.empty:
        chart = model_comparison.set_index("model_name")[["performance_percentage"]]
        st.bar_chart(chart)

    st.markdown("### Hybrid vs Popularity Baseline")
    if metrics.empty:
        st.warning("No original metrics file found.")
    else:
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        chart = metrics.set_index("Metric")[["Hybrid Model", "Popularity Baseline"]]
        st.bar_chart(chart)

    st.markdown("### Controlled Ablation Study")
    st.write(
        "The controlled ablation study addresses the issue of changing multiple parameters "
        "at the same time. In each experiment, only one signal is removed while the remaining "
        "weights are re-normalized."
    )
    display_ablation_table(ablation_results)

    st.markdown("### Main Takeaways")
    st.write("- NDCG@10 is the main metric because ranking position matters in recommendation.")
    st.write("- Recall@10 shows how many clicked articles are retrieved in the top 10.")
    st.write("- Hybrid Model remains the final model because it combines all five recommendation signals.")


def get_options(recs, mode):
    if recs.empty:
        return []
    if mode == "User":
        return recs["user_id"].drop_duplicates().astype(str).tolist()
    return recs["impression_id"].drop_duplicates().astype(str).tolist()


def selected_model_metrics(model_comparison, model_name):
    if model_comparison.empty or "model_name" not in model_comparison.columns:
        return None
    row = model_comparison[model_comparison["model_name"] == model_name]
    if row.empty:
        return None
    return row.iloc[0]


def recommendation_page(recs, model_comparison):
    section_header(
        "Recommendation Demo",
        "Select a user or impression to view ranked news recommendations.",
    )

    if recs.empty:
        st.warning("No recommendation data found.")
        return

    left, right = st.columns([1, 2])

    with left:
        model_name = st.selectbox("Selected Model", list(MODEL_SCORE_COLUMNS.keys()))
        st.caption(MODEL_DESCRIPTIONS[model_name])
        model_metrics = selected_model_metrics(model_comparison, model_name)
        if model_metrics is not None:
            st.metric("Main Metric: NDCG@10", f"{float(model_metrics['ndcg_at_10']):.4f}")
            st.metric("Performance Percentage", f"{float(model_metrics['performance_percentage']):.2f}%")

        mode = st.radio("Search Mode", ["User", "Impression"], horizontal=True)
        options = get_options(recs, mode)
        default = options[0] if options else ""

        selected = st.selectbox(f"Choose {mode.lower()} ID", options[:200], index=0)
        custom = st.text_input(f"Or type a {mode.lower()} ID", value=selected or default)
        top_k = st.slider("Number of recommendations", 5, 20, 10)

        with st.expander("Example IDs"):
            st.write("User IDs:")
            st.code("\n".join(recs["user_id"].drop_duplicates().astype(str).head(6).tolist()))
            st.write("Impression IDs:")
            st.code("\n".join(recs["impression_id"].drop_duplicates().astype(str).head(6).tolist()))

    if mode == "User":
        subset = recs[recs["user_id"].astype(str).str.lower() == custom.lower()]
        if not subset.empty:
            impression = subset.sort_values(["impression_id", "rank"])["impression_id"].iloc[0]
            subset = subset[subset["impression_id"] == impression]
    else:
        subset = recs[recs["impression_id"].astype(str) == custom]

    if subset.empty:
        st.warning("No recommendations found for this selection.")
        return

    subset = rank_for_model(subset, model_name).head(top_k).copy()
    subset["clicked"] = subset.get("label", 0).apply(lambda x: "Clicked" if int(x) == 1 else "Not clicked")

    with right:
        user_id = subset["user_id"].iloc[0] if "user_id" in subset.columns else "N/A"
        impression_id = subset["impression_id"].iloc[0] if "impression_id" in subset.columns else "N/A"
        st.markdown(f"### Recommendations for `{user_id}`")
        st.caption(f"Impression ID: {impression_id}")
        st.write(f"**Selected Model:** {model_name}")

        display_cols = [
            "rank",
            "news_id",
            "title",
            "category",
            "subcategory",
            "score",
            "model_name",
            "clicked",
        ]
        display_cols = [col for col in display_cols if col in subset.columns]
        st.dataframe(subset[display_cols], use_container_width=True, hide_index=True)

    clicked_count = int(subset["label"].sum()) if "label" in subset.columns else 0
    m1, m2, m3 = st.columns(3)
    m1.metric("Displayed Articles", len(subset))
    m2.metric("Clicked in Top-K", clicked_count)
    m3.metric("Best Rank Clicked", int(subset[subset["label"] == 1]["rank"].min()) if clicked_count else "None")

    score_cols = [
        "content_score",
        "cf_score",
        "popularity_score",
        "category_score",
        "subcategory_score",
    ]
    score_cols = [col for col in score_cols if col in subset.columns]

    if score_cols:
        st.markdown("### Score Breakdown")
        st.bar_chart(subset.set_index("news_id")[score_cols])

    st.markdown("### Article Detail")
    detail_subset = subset.copy()
    if "news_id" not in detail_subset.columns:
        st.warning("Article detail is unavailable because news_id is missing.")
        return
    if "title" in detail_subset.columns:
        title_text = detail_subset["title"].fillna("").astype(str).str.strip()
        detail_subset["_article_label"] = title_text.where(
            title_text.ne(""),
            detail_subset["news_id"].astype(str),
        )
    else:
        detail_subset["_article_label"] = detail_subset["news_id"].astype(str)

    selected_label = st.selectbox("Choose an article", detail_subset["_article_label"].tolist())
    selected_row = detail_subset[detail_subset["_article_label"] == selected_label].iloc[0]

    st.write(f"**News ID:** {selected_row.get('news_id', 'N/A')}")
    title_value = selected_row.get("title", "")
    if pd.notna(title_value) and str(title_value).strip():
        st.write(f"**Title:** {title_value}")
    else:
        st.info("Title is not available in the current recommendation file.")
    st.write(f"**Category:** {selected_row.get('category', 'N/A')}")
    st.write(f"**Subcategory:** {selected_row.get('subcategory', 'N/A')}")
    st.write(f"**Rank:** {selected_row.get('rank', 'N/A')}")
    st.write(f"**Final Score:** {score_label(selected_row.get('score', None))}")
    if "abstract" in selected_row and pd.notna(selected_row["abstract"]):
        st.write("**Abstract:**")
        st.write(selected_row["abstract"])


def article_explorer_page(articles, recs=None):
    section_header("Article Explorer", "Browse article metadata used by the recommendation system.")

    if articles.empty and recs is not None and not recs.empty:
        fallback_cols = [
            col
            for col in ["news_id", "category", "subcategory", "title", "abstract"]
            if col in recs.columns
        ]
        if fallback_cols:
            articles = recs[fallback_cols].copy()
            if "news_id" in articles.columns:
                articles = articles.drop_duplicates(subset="news_id")
            else:
                articles = articles.drop_duplicates()

    if articles.empty:
        st.warning("No article data found.")
        st.write(
            "Place MINDsmall_train/news.tsv and MINDsmall_dev/news.tsv beside app.py, "
            "or provide data/articles_sample.csv for the deployed demo."
        )
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        categories = ["All"] + sorted(articles["category"].dropna().astype(str).unique().tolist()) if "category" in articles.columns else ["All"]
        category = st.selectbox("Category", categories)
        keyword = st.text_input("Search title keyword")

    filtered = articles.copy()
    if category != "All" and "category" in filtered.columns:
        filtered = filtered[filtered["category"].astype(str) == category]
    if keyword:
        if "title" in filtered.columns:
            filtered = filtered[filtered["title"].astype(str).str.contains(keyword, case=False, na=False)]
        elif "news_id" in filtered.columns:
            filtered = filtered[filtered["news_id"].astype(str).str.contains(keyword, case=False, na=False)]

    with col2:
        st.metric("Matching Articles", f"{len(filtered):,}")

    show_cols = [col for col in ["news_id", "category", "subcategory", "title", "abstract"] if col in filtered.columns]
    st.dataframe(filtered[show_cols].head(100), use_container_width=True, hide_index=True)


def custom_news_ranking_page(recs):
    section_header(
        "Custom News Ranking",
        "Input a new article and insert it into an existing recommendation list.",
    )

    if recs.empty:
        st.warning("No recommendation data found.")
        return

    st.info(
        "This page tests how a newly typed article would be ranked inside an existing candidate list. "
        "Choose a user or impression first, then enter the external article title, abstract, category, "
        "and subcategory. The popularity score is estimated automatically from historical data."
    )

    default_example = EXTERNAL_NEWS_EXAMPLES["AI study tool"]
    st.session_state.setdefault("custom_title", default_example["title"])
    st.session_state.setdefault("custom_abstract", default_example["abstract"])
    st.session_state.setdefault("custom_category", default_example["category"])
    st.session_state.setdefault("custom_subcategory", default_example["subcategory"])

    left, right = st.columns([1, 2])
    with left:
        model_name = st.selectbox("Selected Model", list(MODEL_SCORE_COLUMNS.keys()))
        st.caption(MODEL_DESCRIPTIONS[model_name])
        mode = st.radio("Context Mode", ["User", "Impression"], horizontal=True)
        options = get_options(recs, mode)
        selected = st.selectbox(f"Choose {mode.lower()} ID", options[:200], index=0)
        custom_id = st.text_input(f"Or type a {mode.lower()} ID", value=selected)
        top_k = st.slider("Display top recommendations", 5, 20, 10)

        st.markdown("### New Article")
        with st.expander("Example external news", expanded=True):
            example_names = list(EXTERNAL_NEWS_EXAMPLES.keys())
            example_cols = st.columns(2)
            for idx, example_name in enumerate(example_names):
                if example_cols[idx % 2].button(example_name, use_container_width=True):
                    example = EXTERNAL_NEWS_EXAMPLES[example_name]
                    st.session_state["custom_title"] = example["title"]
                    st.session_state["custom_abstract"] = example["abstract"]
                    st.session_state["custom_category"] = example["category"]
                    st.session_state["custom_subcategory"] = example["subcategory"]
                    st.session_state.pop("custom_news_payload", None)

        title = st.text_input("Title", key="custom_title")
        abstract = st.text_area("Abstract", key="custom_abstract", height=130)
        category = st.text_input("Category", key="custom_category")
        subcategory = st.text_input("Subcategory", key="custom_subcategory")

        with st.expander("What should I enter?"):
            st.write("Use a normal news headline and a short abstract, similar to the MIND news data.")
            st.code(
                "Title: New AI tool helps students summarize daily news faster\n"
                "Abstract: A newly released AI tool helps students read, summarize, "
                "and compare news articles more efficiently.\n"
                "Category: news\n"
                "Subcategory: technology"
            )
            st.write(
                "The article is inserted as CUSTOM_NEWS. The system then recalculates the score "
                "and sorts it together with the original candidate articles. Popularity is not "
                "entered manually; it is estimated from similar historical articles."
            )

        current_payload = (
            model_name,
            mode,
            custom_id,
            top_k,
            title.strip(),
            abstract.strip(),
            category.strip(),
            subcategory.strip(),
        )
        analyze_clicked = st.button("Analyze / Re-rank", type="primary", use_container_width=True)
        if analyze_clicked:
            st.session_state["custom_news_payload"] = current_payload

    if not title.strip():
        st.warning("Please enter a title for the external news article.")
        return
    if not abstract.strip():
        st.warning("Please enter an abstract for the external news article.")
        return
    if not category.strip():
        category = "unknown"
    if not subcategory.strip():
        subcategory = "unknown"

    if st.session_state.get("custom_news_payload") != current_payload:
        with right:
            st.markdown("### Re-ranked Candidate List")
            st.info("Enter or choose an external news article, then click Analyze / Re-rank.")
        return

    if mode == "User":
        context = recs[recs["user_id"].astype(str).str.lower() == custom_id.lower()]
        if not context.empty:
            impression = context.sort_values(["impression_id", "rank"])["impression_id"].iloc[0]
            context = context[context["impression_id"] == impression]
    else:
        context = recs[recs["impression_id"].astype(str) == custom_id]

    if context.empty:
        st.warning("No recommendation context found for this selection.")
        return

    context = context.sort_values("rank").copy()
    scores, popularity_source = custom_news_scores(context, title, abstract, category, subcategory, recs)
    scores["score"] = custom_score_for_model(scores, model_name)

    custom_row = {
        "impression_id": context["impression_id"].iloc[0] if "impression_id" in context.columns else "custom",
        "user_id": context["user_id"].iloc[0] if "user_id" in context.columns else custom_id,
        "news_id": "CUSTOM_NEWS",
        "source": "External Input",
        "title": title,
        "abstract": abstract,
        "category": category,
        "subcategory": subcategory,
        "label": 0,
        **scores,
    }

    context = rank_for_model(context, model_name)
    context["source"] = "Dataset"
    ranked = pd.concat([context, pd.DataFrame([custom_row])], ignore_index=True, sort=False)
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["model_name"] = model_name
    shown = ranked.head(top_k).copy()
    shown["clicked"] = shown.get("label", 0).apply(lambda x: "Clicked" if int(x) == 1 else "Not clicked")

    with right:
        st.markdown("### Re-ranked Candidate List")
        st.caption("The custom article is inserted as `CUSTOM_NEWS` and ranked by the selected model.")
        st.write(f"**Selected Model:** {model_name}")
        display_cols = [
            "rank",
            "source",
            "news_id",
            "title",
            "category",
            "subcategory",
            "score",
            "model_name",
            "clicked",
        ]
        display_cols = [col for col in display_cols if col in shown.columns]
        st.dataframe(shown[display_cols], use_container_width=True, hide_index=True)

        custom_rank = int(ranked[ranked["news_id"] == "CUSTOM_NEWS"]["rank"].iloc[0])
        custom_score = float(ranked[ranked["news_id"] == "CUSTOM_NEWS"]["score"].iloc[0])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Custom News Rank", custom_rank)
        c2.metric("Custom News Score", f"{custom_score:.4f}")
        c3.metric("Candidate List Size", len(ranked))
        c4.metric("Auto Popularity", f"{scores['popularity_score']:.4f}")
        st.caption(popularity_source)

    score_cols = [
        "content_score",
        "cf_score",
        "popularity_score",
        "category_score",
        "subcategory_score",
    ]
    score_cols = [col for col in score_cols if col in shown.columns]
    if score_cols:
        st.markdown("### Score Breakdown")
        chart_data = shown.set_index("news_id")[score_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        st.bar_chart(chart_data)

    st.markdown("### How the Custom Score Is Estimated")
    st.write(
        "For deployed use, the app uses a lightweight approximation so it can run without the full "
        "embedding matrix. It compares the new title and abstract with the selected user's current "
        "candidate context, checks category and subcategory match, and automatically estimates "
        "popularity from historical articles with the same subcategory or category. Collaborative "
        "filtering is set to 0 because a new external article has no historical click behavior."
    )
    if model_name == "Popularity-based Model":
        st.warning("External news has no real click history, so its popularity is estimated automatically from similar historical articles.")
    elif model_name == "Collaborative Filtering Model":
        st.warning("External news has no interaction history, so collaborative filtering score is unavailable and set to 0.")


def about_page():
    section_header("About the Model")

    st.markdown(
        """
        This application demonstrates a candidate-based personalized news recommendation system.
        In each impression, the user is shown a fixed set of candidate news articles, and the model
        reorders those candidates so that articles the user is more likely to click appear higher.

        **System task**  
        The task is not to classify a single article as good or bad. The task is to rank many
        candidate articles for a specific user. For that reason, the output is an ordered list:
        rank 1 is the strongest recommendation, followed by rank 2, rank 3, and so on.

        **Data used by the system**  
        The system uses MINDsmall news metadata and user behavior logs. The news file provides
        article ID, category, subcategory, title, and abstract. The behavior file provides user ID,
        reading history, impression ID, candidate articles, and click labels. The app can also
        load the original MINDsmall_train/news.tsv and MINDsmall_dev/news.tsv files to recover
        article titles and abstracts for browsing.

        **Text representation**  
        Article title and abstract are combined into one text field. In the full notebook, this text
        is converted into TF-IDF vectors and sentence embeddings. TF-IDF captures exact keywords,
        while sentence embeddings capture semantic similarity. The stored vectors are used to compare
        user interest profiles with candidate articles.

        **Final hybrid signals**
        - **Content similarity:** compares article text with the user's reading interests.
        - **Collaborative filtering:** uses behavior patterns from similar users.
        - **Popularity:** gives a prior score to articles that are generally clicked more often.
        - **Category preference:** matches broad user interests such as news, sports, or finance.
        - **Subcategory preference:** matches more specific interests such as football, weather, or technology.

        **Final scoring formula**  
        The final model uses a weighted score:
        `final_score = 0.45*content + 0.10*cf + 0.20*popularity + 0.10*category + 0.15*subcategory`.
        After every candidate receives a score, the candidates are sorted from highest score to lowest score.

        **Why the Hybrid Model is kept**  
        Single-signal models are easier to explain, but they only use one view of the problem.
        Content-based ranking can match text but may ignore general news popularity. Popularity-based
        ranking is stable but weakly personalized. Collaborative filtering can capture user behavior
        but struggles when interaction history is sparse. The Hybrid Model is kept because it combines
        these strengths and achieved the best NDCG@10 in the controlled model comparison.

        **How the controlled comparison issue is handled**  
        The model comparison now separates complete model strategies: Hybrid Model, Content-based Model,
        Popularity-based Model, and Collaborative Filtering Model. The ablation study then starts from
        the final Hybrid Model and removes only one signal at a time. This makes the experiment easier
        to interpret because each ablation result shows the effect of one missing signal.

        **Why not accuracy?**  
        Accuracy is suitable for classification, but this project is a ranking task. If a clicked article
        appears at rank 1, the recommendation is much better than if the same clicked article appears
        at rank 10. Accuracy does not capture this ranking position.

        **Why these metrics?**
        - **Precision@K:** how many top-K recommendations were clicked.
        - **Recall@K:** how many clicked articles were retrieved in the top-K list.
        - **NDCG@K:** whether clicked articles were placed near the top of the ranking.

        **External news input**  
        A newly typed article has no historical clicks and no similar-user interaction history.
        Therefore, collaborative filtering is unavailable for that article and is set to 0 in the demo.
        The custom article can still be ranked using content similarity, category match,
        subcategory match, and an automatically estimated popularity value based on similar historical
        articles. The user cannot manually control this popularity score.
        """
    )

    weights = pd.DataFrame(
        [
            ["Content similarity", 0.45],
            ["Collaborative filtering", 0.10],
            ["Popularity", 0.20],
            ["Category preference", 0.10],
            ["Subcategory preference", 0.15],
        ],
        columns=["Signal", "Weight"],
    )
    st.dataframe(weights, use_container_width=True, hide_index=True)


def main():
    articles, recs, metrics, articles_path, recs_path, metrics_path = prepare_data()
    model_comparison, _ = load_model_comparison()
    ablation_results, _ = load_ablation_results()

    if model_comparison.empty and not recs.empty:
        model_comparison = build_model_comparison(recs)
    if ablation_results.empty and not recs.empty:
        ablation_results = build_ablation_results(recs)

    sidebar_controls(articles, recs, metrics_path, recs_path, articles_path)

    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Evaluation", "Recommendation Demo", "Custom News Ranking", "Article Explorer", "About"],
    )

    if page == "Home":
        home_page(articles, recs, metrics)
    elif page == "Evaluation":
        evaluation_page(metrics, model_comparison, ablation_results)
    elif page == "Recommendation Demo":
        recommendation_page(recs, model_comparison)
    elif page == "Custom News Ranking":
        custom_news_ranking_page(recs)
    elif page == "Article Explorer":
        article_explorer_page(articles, recs)
    else:
        about_page()


if __name__ == "__main__":
    main()
