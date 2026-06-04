from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

ARTICLE_FILES = [
    BASE_DIR / "articles.csv",
    BASE_DIR / "articles_proposal_cleaned.csv",
    DATA_DIR / "demo_articles.csv",
    DATA_DIR / "articles_sample.csv",
    DATA_DIR / "articles.csv",
]

RECOMMENDATION_FILES = [
    BASE_DIR / "ranked_recommendations.csv",
    BASE_DIR / "proposal_improved_ranked_recommendations.csv",
    BASE_DIR / "proposal_ranked_recommendations.csv",
    DATA_DIR / "demo_ranked_recommendations.csv",
    DATA_DIR / "ranked_recommendations.csv",
]

METRICS_FILES = [
    BASE_DIR / "metrics_comparison.csv",
    BASE_DIR / "proposal_improved_metrics_comparison.csv",
    BASE_DIR / "proposal_metrics_comparison.csv",
    RESULTS_DIR / "metrics_comparison.csv",
]


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
    if path is None:
        return pd.DataFrame(), None
    return pd.read_csv(path), path


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


def evaluation_page(metrics):
    section_header(
        "Evaluation Results",
        "Recommendation quality is evaluated using Precision@K, Recall@K, and NDCG@K.",
    )

    if metrics.empty:
        st.warning("No metrics file found.")
        return

    st.dataframe(metrics, use_container_width=True, hide_index=True)

    chart = metrics.set_index("Metric")[["Hybrid Model", "Popularity Baseline"]]
    st.bar_chart(chart)

    st.markdown("### Main Takeaways")
    st.write("- NDCG@10 is the main metric because ranking position matters in recommendation.")
    st.write("- Recall@10 shows how many clicked articles are retrieved in the top 10.")
    st.write("- The popularity baseline is strong, but it does not personalize recommendations.")


def get_options(recs, mode):
    if recs.empty:
        return []
    if mode == "User":
        return recs["user_id"].drop_duplicates().astype(str).tolist()
    return recs["impression_id"].drop_duplicates().astype(str).tolist()


def recommendation_page(recs):
    section_header(
        "Recommendation Demo",
        "Select a user or impression to view ranked news recommendations.",
    )

    if recs.empty:
        st.warning("No recommendation data found.")
        return

    left, right = st.columns([1, 2])

    with left:
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

    subset = subset.sort_values("rank").head(top_k).copy()
    subset["clicked"] = subset.get("label", 0).apply(lambda x: "Clicked" if int(x) == 1 else "Not clicked")

    with right:
        user_id = subset["user_id"].iloc[0] if "user_id" in subset.columns else "N/A"
        impression_id = subset["impression_id"].iloc[0] if "impression_id" in subset.columns else "N/A"
        st.markdown(f"### Recommendations for `{user_id}`")
        st.caption(f"Impression ID: {impression_id}")

        display_cols = [
            "rank",
            "news_id",
            "title",
            "category",
            "subcategory",
            "score",
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
    article_titles = subset["title"].fillna(subset["news_id"]).tolist() if "title" in subset.columns else subset["news_id"].tolist()
    selected_title = st.selectbox("Choose an article", article_titles)
    selected_row = subset[subset["title"].fillna(subset["news_id"]) == selected_title].iloc[0]

    st.write(f"**News ID:** {selected_row.get('news_id', 'N/A')}")
    st.write(f"**Category:** {selected_row.get('category', 'N/A')}")
    st.write(f"**Subcategory:** {selected_row.get('subcategory', 'N/A')}")
    st.write(f"**Rank:** {selected_row.get('rank', 'N/A')}")
    st.write(f"**Final Score:** {score_label(selected_row.get('score', None))}")
    if "abstract" in selected_row and pd.notna(selected_row["abstract"]):
        st.write("**Abstract:**")
        st.write(selected_row["abstract"])


def article_explorer_page(articles):
    section_header("Article Explorer", "Browse article metadata used by the recommendation system.")

    if articles.empty:
        st.warning("No article data found.")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        categories = ["All"] + sorted(articles["category"].dropna().astype(str).unique().tolist()) if "category" in articles.columns else ["All"]
        category = st.selectbox("Category", categories)
        keyword = st.text_input("Search title keyword")

    filtered = articles.copy()
    if category != "All" and "category" in filtered.columns:
        filtered = filtered[filtered["category"].astype(str) == category]
    if keyword and "title" in filtered.columns:
        filtered = filtered[filtered["title"].astype(str).str.contains(keyword, case=False, na=False)]

    with col2:
        st.metric("Matching Articles", f"{len(filtered):,}")

    show_cols = [col for col in ["news_id", "category", "subcategory", "title", "abstract"] if col in filtered.columns]
    st.dataframe(filtered[show_cols].head(100), use_container_width=True, hide_index=True)


def about_page():
    section_header("About the Model")

    st.markdown(
        """
        This application demonstrates a candidate-based personalized news recommendation system.

        **Why not accuracy?**  
        Accuracy is suitable for classification, but news recommendation is a ranking task.
        The goal is to place clicked articles near the top of the ranked list.

        **Why these metrics?**
        - **Precision@K:** how many top-K recommendations were clicked.
        - **Recall@K:** how many clicked articles were retrieved in top-K.
        - **NDCG@K:** how highly clicked articles were ranked.

        **Final hybrid signals**
        - content similarity
        - collaborative filtering
        - popularity
        - category preference
        - subcategory preference
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
    sidebar_controls(articles, recs, metrics_path, recs_path, articles_path)

    page = st.sidebar.radio(
        "Navigation",
        ["Home", "Evaluation", "Recommendation Demo", "Article Explorer", "About"],
    )

    if page == "Home":
        home_page(articles, recs, metrics)
    elif page == "Evaluation":
        evaluation_page(metrics)
    elif page == "Recommendation Demo":
        recommendation_page(recs)
    elif page == "Article Explorer":
        article_explorer_page(articles)
    else:
        about_page()


if __name__ == "__main__":
    main()
