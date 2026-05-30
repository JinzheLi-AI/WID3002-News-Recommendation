import numpy as np
import pandas as pd
import pickle
import json

BASE_DIR = r"C:\Users\Administrator\OneDrive\Desktop\WID3002 MINDsmall Dataset"

print("Loading model files ...")

articles         = pd.read_csv(BASE_DIR + r"\data\articles.csv")
article_embeddings = np.load(BASE_DIR + r"\models\article_embeddings.npy")

with open(BASE_DIR + r"\data\article_id_index.json") as f:
    article_id_index = json.load(f)

with open(BASE_DIR + r"\models\user_profiles.pkl", "rb") as f:
    user_profiles = pickle.load(f)

with open(BASE_DIR + r"\data\popular_by_category.json") as f:
    popular_by_category = json.load(f)

interactions = pd.read_csv(BASE_DIR + r"\data\ranked_recommendations.csv")

print("All files loaded.\n")

# ── Scoring functions ──────────────────────────────────────────────────────
def content_score(user_vector, article_embs):
    norm_user = user_vector / (np.linalg.norm(user_vector) + 1e-9)
    norms     = np.linalg.norm(article_embs, axis=1, keepdims=True) + 1e-9
    return article_embs.dot(norm_user) / norms.squeeze()

def recommend(user_id, top_k=10, alpha=0.6):
    profile = user_profiles.get(user_id, None)

    # Cold-start fallback
    if profile is None:
        all_popular = [nid for nids in popular_by_category.values() for nid in nids]
        rec_ids = list(dict.fromkeys(all_popular))[:top_k]
    else:
        scores  = content_score(profile, article_embeddings)
        top_idx = np.argsort(scores)[::-1][:top_k]
        rec_ids = [articles.iloc[i]["news_id"] for i in top_idx]

    results = []
    for nid in rec_ids:
        row = articles[articles["news_id"] == nid]
        if not row.empty:
            r = row.iloc[0]
            results.append({"news_id": nid, "category": r["category"], "title": r["title"]})
    return results

# ── Run ────────────────────────────────────────────────────────────────────
user_id = input("Enter user ID (e.g. U100): ").strip()
recs    = recommend(user_id)

if not recs:
    print("No recommendations found.")
else:
    print(f"\nTop {len(recs)} recommendations for {user_id}:\n")
    for i, r in enumerate(recs, 1):
        print(f"  {i}. [{r['category']}] {r['title']}")
