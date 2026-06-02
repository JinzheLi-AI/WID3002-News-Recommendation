from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

ARTICLE_FILES = [
    DATA_DIR / "demo_articles.csv",
    DATA_DIR / "articles_sample.csv",
    DATA_DIR / "articles.csv",
]

RECOMMENDATION_FILES = [
    DATA_DIR / "demo_ranked_recommendations.csv",
    DATA_DIR / "ranked_recommendations.csv",
]

METRICS_FILE = RESULTS_DIR / "metrics_comparison.csv"


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def load_data():
    articles_path = first_existing(ARTICLE_FILES)
    recs_path = first_existing(RECOMMENDATION_FILES)

    if articles_path is None:
        raise FileNotFoundError("No article data found in data/.")
    if recs_path is None:
        raise FileNotFoundError("No recommendation data found in data/.")

    articles = pd.read_csv(articles_path)
    recs = pd.read_csv(recs_path)

    article_cols = [col for col in ["news_id", "category", "subcategory", "title"] if col in articles.columns]
    recs = recs.merge(articles[article_cols], on="news_id", how="left")
    return recs


def show_metrics():
    if not METRICS_FILE.exists():
        print("\nNo metrics file found. Run the notebook first to generate results.")
        return

    metrics = pd.read_csv(METRICS_FILE)
    print("\nFinal Evaluation Results")
    print(metrics.to_string(index=False))

    ndcg = metrics[metrics["Metric"] == "NDCG@10"]
    if not ndcg.empty:
        hybrid = float(ndcg["Hybrid Model"].iloc[0])
        baseline = float(ndcg["Popularity Baseline"].iloc[0])
        improvement = hybrid - baseline
        relative = improvement / baseline * 100 if baseline else 0

        print("\nMain Result")
        print(f"Hybrid NDCG@10: {hybrid:.4f}")
        print(f"Baseline NDCG@10: {baseline:.4f}")
        print(f"Improvement: {improvement:.4f} ({relative:.1f}%)")


def show_examples(recs):
    impression_examples = recs["impression_id"].drop_duplicates().head(5).tolist()
    user_examples = recs["user_id"].drop_duplicates().head(5).tolist()

    print("\nExample impression_id values")
    for value in impression_examples:
        print(f"  {value}")

    print("\nExample user_id values")
    for value in user_examples:
        print(f"  {value}")


def show_score_explanation():
    print("\nScore Columns")
    print("  score              final hybrid ranking score")
    print("  content_score      semantic match between user profile and article")
    print("  cf_score           collaborative filtering signal from similar users")
    print("  popularity_score   article click popularity from training data")
    print("  category_score     broad topic match from user history")
    print("  subcategory_score  specific topic match from user history")
    print("  label              1 means clicked in validation data, 0 means not clicked")


def print_recommendations(subset, title, top_k=10):
    subset = subset.sort_values("rank").head(top_k)
    if subset.empty:
        print("No recommendations found.")
        return

    print(f"\n{title}\n")
    for _, row in subset.iterrows():
        clicked = "clicked" if int(row.get("label", 0)) == 1 else "not clicked"
        category = row.get("category", "unknown")
        subcategory = row.get("subcategory", "")
        label = f"{category}/{subcategory}" if isinstance(subcategory, str) and subcategory else category
        title = row.get("title", "Title not found")
        score = float(row.get("score", 0))

        print(f"{int(row['rank']):>2}. [{label}] {title}")
        print(f"    score={score:.4f} | {clicked}")


def search_by_impression(recs):
    show_examples(recs)
    impression_id = input("\nEnter impression_id: ").strip()
    subset = recs[recs["impression_id"].astype(str) == impression_id]

    if subset.empty:
        print(f"No recommendations found for impression_id={impression_id}")
        return

    user_id = subset["user_id"].iloc[0]
    print_recommendations(subset, f"Top recommendations for impression {impression_id} ({user_id})")


def search_by_user(recs):
    show_examples(recs)
    user_id = input("\nEnter user_id: ").strip()
    subset = recs[recs["user_id"].astype(str).str.lower() == user_id.lower()]

    if subset.empty:
        print(f"No recommendations found for user_id={user_id}")
        return

    first_impression = subset.sort_values(["impression_id", "rank"])["impression_id"].iloc[0]
    subset = subset[subset["impression_id"] == first_impression]
    print_recommendations(subset, f"Top recommendations for user {user_id} (impression {first_impression})")


def show_top_clicked_predictions(recs, top_k=10):
    if "label" not in recs.columns:
        print("No label column found.")
        return

    clicked = recs[recs["label"] == 1].sort_values("rank").head(top_k)
    print_recommendations(clicked, "Clicked articles ranked by the model", top_k=top_k)


def menu():
    recs = load_data()

    while True:
        print("\nPersonalized News Recommendation Demo")
        print("1. Show final evaluation results")
        print("2. Show example IDs")
        print("3. Search recommendations by impression_id")
        print("4. Search recommendations by user_id")
        print("5. Show clicked articles ranked by the model")
        print("6. Explain score columns")
        print("0. Exit")

        choice = input("Choose option: ").strip()

        if choice == "1":
            show_metrics()
        elif choice == "2":
            show_examples(recs)
        elif choice == "3":
            search_by_impression(recs)
        elif choice == "4":
            search_by_user(recs)
        elif choice == "5":
            show_top_clicked_predictions(recs)
        elif choice == "6":
            show_score_explanation()
        elif choice == "0":
            break
        else:
            print("Invalid option.")


if __name__ == "__main__":
    menu()
