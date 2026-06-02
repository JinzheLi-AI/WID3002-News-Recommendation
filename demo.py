from pathlib import Path

import pandas as pd


BASE_DIR = Path(r"C:\Users\Administrator\OneDrive\Desktop\WID3002 DataSet")

METRICS_FILE = BASE_DIR / "metrics_comparison.csv"
RECOMMENDATIONS_FILE = BASE_DIR / "ranked_recommendations.csv"


def show_metrics():
    if not METRICS_FILE.exists():
        print("metrics_comparison.csv was not found.")
        print("Run the notebook first to generate the evaluation results.")
        return

    metrics = pd.read_csv(METRICS_FILE)
    print("\nFinal Evaluation Results")
    print(metrics.to_string(index=False))

    ndcg_row = metrics[metrics["Metric"] == "NDCG@10"]
    if not ndcg_row.empty:
        hybrid = float(ndcg_row["Hybrid Model"].iloc[0])
        baseline = float(ndcg_row["Popularity Baseline"].iloc[0])
        improvement = hybrid - baseline
        relative = improvement / baseline * 100 if baseline else 0

        print("\nMain Result")
        print(f"Hybrid NDCG@10: {hybrid:.4f}")
        print(f"Baseline NDCG@10: {baseline:.4f}")
        print(f"Absolute improvement: {improvement:.4f}")
        print(f"Relative improvement: {relative:.1f}%")


def show_sample_recommendations(n=10):
    if not RECOMMENDATIONS_FILE.exists():
        print("\nranked_recommendations.csv was not found.")
        return

    usecols = [
        "impression_id",
        "user_id",
        "news_id",
        "rank",
        "label",
        "score",
        "content_score",
        "popularity_score",
        "category_score",
        "subcategory_score",
    ]

    recommendations = pd.read_csv(RECOMMENDATIONS_FILE, usecols=usecols)
    first_impression = recommendations["impression_id"].iloc[0]
    sample = (
        recommendations[recommendations["impression_id"] == first_impression]
        .sort_values("rank")
        .head(n)
    )

    print(f"\nSample ranked recommendations for impression {first_impression}")
    print(sample.to_string(index=False))


if __name__ == "__main__":
    show_metrics()
    show_sample_recommendations()
