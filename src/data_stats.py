"""
data_stats.py
Purpose: compute the exploratory statistics of Chapter 3 (Tables 3.3, 3.4, Figures 3.1 - 3.3)
         directly from MovieLens, so the web app shows exactly the numbers in the report.
         The result is saved to cache/data_stats.json and simply read back afterwards.

Try it:  python -m src.data_stats
"""
import json

import numpy as np
import pandas as pd

from src import config
from src.data_loader import build_matrix, load_movies, load_ratings, split_train_test

STATS_FILE = config.CACHE_DIR / "data_stats.json"


def _describe(counts: np.ndarray) -> dict:
    return {"min": int(counts.min()), "median": float(np.median(counts)),
            "mean": round(float(counts.mean()), 1), "max": int(counts.max())}


def _share_buckets(values: np.ndarray, edges: list) -> list:
    """edges = [(label, low, high)], high = None means no upper bound."""
    total = len(values)
    out = []
    for label, low, high in edges:
        mask = (values >= low) if high is None else ((values >= low) & (values <= high))
        n = int(mask.sum())
        out.append({"label": label, "count": n, "share": round(100 * n / total, 1)})
    return out


def compute_stats() -> dict:
    ratings = load_ratings()
    movies = load_movies()
    n_ratings = len(ratings)
    n_users = ratings["userId"].nunique()
    n_items = ratings["movieId"].nunique()
    density = n_ratings / (n_users * n_items)

    per_user = ratings.groupby("userId").size().to_numpy()
    per_movie = ratings.groupby("movieId").size().to_numpy()
    sorted_movie = np.sort(per_movie)[::-1]
    top10 = int(round(len(sorted_movie) * 0.1))
    times = pd.to_datetime(ratings["timestamp"], unit="s")

    # Figure 3.1: rating distribution
    dist = ratings["rating"].value_counts().sort_index()
    rating_dist = [{"rating": float(r), "count": int(c), "share": round(100 * c / n_ratings, 1)}
                   for r, c in dist.items()]

    # Figure 3.2: long tail, movies grouped by number of ratings
    long_tail = _share_buckets(per_movie, [
        ("1", 1, 1), ("2", 2, 2), ("3–5", 3, 5), ("6–10", 6, 10),
        ("11–20", 11, 20), ("21–50", 21, 50), ("51–100", 51, 100), ("> 100", 101, None)])

    # Figure 3.3: shared raters per movie pair, on the training set
    train_df, test_df = split_train_test(ratings)
    rm = build_matrix(train_df, tag="train")
    C = rm.M.T @ rm.M
    np.fill_diagonal(C, 0)
    support = C[C > 0]                      # symmetric matrix: each pair appears twice
    del C
    support_buckets = _share_buckets(support, [
        ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4),
        ("5–9", 5, 9), ("10–19", 10, 19), ("≥ 20", 20, None)])
    unseen = int((~test_df["movieId"].isin(train_df["movieId"])).sum())

    return {
        "overview": {
            "n_ratings": n_ratings, "n_users": int(n_users), "n_movies_rated": int(n_items),
            "n_movies_catalog": int(len(movies)), "density": round(100 * density, 2),
            "rating_min": float(ratings["rating"].min()), "rating_max": float(ratings["rating"].max()),
            "rating_mean": round(float(ratings["rating"].mean()), 2),
            "share_ge3": round(100 * float((ratings["rating"] >= 3).mean()), 1),
            "time_start": times.min().strftime("%m/%Y"), "time_end": times.max().strftime("%m/%Y"),
        },
        "per_user": _describe(per_user),
        "per_movie": _describe(per_movie),
        "rating_dist": rating_dist,
        "long_tail": {
            "buckets": long_tail,
            "top10_movies": top10,
            "top10_share": round(100 * float(sorted_movie[:top10].sum()) / n_ratings, 1),
            "single": int((per_movie == 1).sum()),
            "single_share": round(100 * float((per_movie == 1).mean()), 1),
            "le5_share": round(100 * float((per_movie <= 5).mean()), 1),
        },
        "support": {
            "pairs": int(len(support) // 2),
            "buckets": [dict(b, count=b["count"] // 2) for b in support_buckets],
        },
        "split": {
            "train": len(train_df), "test": len(test_df),
            "train_users": int(rm.shape[0]), "train_movies": int(rm.shape[1]),
            "test_unseen_rows": unseen, "test_unseen_share": round(100 * unseen / len(test_df), 1),
        },
    }


def load_stats() -> dict:
    """Read from the cache if present (and built from the same number of ratings); otherwise recompute and save."""
    if STATS_FILE.exists():
        stats = json.loads(STATS_FILE.read_text(encoding="utf-8"))
        if stats.get("overview", {}).get("n_ratings") == len(load_ratings()):
            return stats
    stats = compute_stats()
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


if __name__ == "__main__":
    s = compute_stats()
    print(json.dumps(s["overview"], ensure_ascii=False, indent=1))
    print("support:", s["support"])
    print("long tail:", s["long_tail"])
