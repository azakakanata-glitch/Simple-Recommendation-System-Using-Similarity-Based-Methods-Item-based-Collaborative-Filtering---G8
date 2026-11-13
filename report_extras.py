"""
report_extras.py
Purpose: compute the report numbers that no other part of the project produces:
  Chapter 3 - number of rows of tags.csv (Table 3.3); movie pairs with Cosine close to 1 and
              how many of them rest on a single user (Section 3.2); redraws Figures 3.1, 3.2, 3.3.
  Chapter 5 - Table 5.2 (performance), Table 5.4 (popularity of recommended movies),
              Table 5.5 (first 5 recommendations for profile 1 with Cosine and Pearson).

This file does NOT modify any other file. It only calls existing functions in src/ and
evaluate.py, and only writes new files starting with "extra_" to the results/ folder.
Cache files in cache/ may be read, or new ones added with the current naming rule.

How to run (from the project root folder):
    python report_extras.py              # run everything (about 30-60 seconds)
    python report_extras.py --no-perf    # skip the performance part (Table 5.2)
"""
import argparse
import io
import time
import tracemalloc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy import sparse

from evaluate import TestSet, rating_metrics, topn_metrics
from src import config
from src.data_loader import build_matrix, load_movies, load_ratings, split_train_test
from src.recommender import predict_all, rank_scores, recommend_top_n
from src.similarity import build_similarity, co_counts, cosine

OUT = config.RESULTS_DIR
PREFIX = "extra_"

# ---------------- number formatting ----------------
def vn(x, d=1):
    """1234.5 -> '1,234.5'"""
    return f"{x:,.{d}f}"


def vn_int(x):
    return vn(x, 0)


def save_csv(df: pd.DataFrame, name: str):
    path = OUT / f"{PREFIX}{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")   # utf-8-sig so that Excel reads special characters correctly
    return path


def header(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------- chart style (same as the report figures) ----------------
BLUE, INK, MUTED = "#2a78d6", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8, "axes.axisbelow": True,
})


# =====================================================================
# CHAPTER 3
# =====================================================================
def chapter3(ratings: pd.DataFrame, rm_train):
    header("CHAPTER 3. DATA STATISTICS")
    rows = []

    # Table 3.3: number of rows of tags.csv
    tags_path = config.DATA_DIR / "tags.csv"
    if tags_path.exists():
        n_tags = len(pd.read_csv(tags_path))
        print(f"tags.csv: {vn_int(n_tags)} rows")
        rows.append({"metric": "Rows in tags.csv (Table 3.3)", "value": n_tags})
    else:
        print("data/tags.csv not found, skipped (this file is optional).")

    # Section 3.2: movie pairs with Cosine close to 1 and their number of shared raters
    S = cosine(rm_train)                     # full Cosine, no shrinkage, no top-k
    C = co_counts(rm_train)
    near_one = S >= 0.999
    n_near = int(near_one.sum() // 2)        # symmetric matrix: each pair counted twice
    n_near_single = int((near_one & (C == 1)).sum() // 2)
    n_pairs = int((C > 0).sum() // 2)
    n_single = int((C == 1).sum() // 2)
    del S, C, near_one
    print(f"Movie pairs with at least 1 shared rater (training set): {vn_int(n_pairs)}")
    print(f"  of which exactly 1 shared rater: {vn_int(n_single)} ({vn(100 * n_single / n_pairs)}%)")
    print(f"Movie pairs with Cosine >= 0.999: {vn_int(n_near)}")
    print(f"  of which based on a single user: {vn_int(n_near_single)} ({vn(100 * n_near_single / n_near)}%)")
    rows += [
        {"metric": "Movie pairs with a shared rater (train)", "value": n_pairs},
        {"metric": "Movie pairs with exactly 1 shared rater", "value": n_single},
        {"metric": "Movie pairs with Cosine >= 0.999", "value": n_near},
        {"metric": "Pairs with Cosine >= 0.999 based on 1 user", "value": n_near_single},
    ]
    print("Saved:", save_csv(pd.DataFrame(rows), "chapter3_stats").name)

    # Figure 3.1: rating distribution
    vc = ratings["rating"].value_counts().sort_index()
    share = vc / vc.sum() * 100
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    ax.bar([vn(x) for x in vc.index], vc.values, color=BLUE, width=0.72)
    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Number of ratings")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: vn_int(v)))
    ax.grid(axis="x", visible=False)
    for i in np.argsort(-vc.values)[:2]:     # label the 2 most common ratings
        ax.annotate(f"{vn(vc.index[i])} stars: {vn(share.iloc[i])}%", (i, vc.values[i]),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.text(0.01, 0.97, f"Mean {vn(ratings['rating'].mean(), 2)}; "
            f"{vn(100 * (ratings['rating'] >= 3).mean())}% of ratings are 3.0 stars or higher",
            transform=ax.transAxes, va="top", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / f"{PREFIX}fig3_1_rating_distribution.png")
    plt.close(fig)

    # Figure 3.2: long tail of ratings per movie
    per_movie = ratings.groupby("movieId").size().sort_values(ascending=False).to_numpy()
    x = np.arange(1, len(per_movie) + 1)
    top10 = int(round(len(per_movie) * 0.1))
    top10_share = 100 * per_movie[:top10].sum() / per_movie.sum()
    n_one = int((per_movie == 1).sum())
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    ax.fill_between(x, per_movie, color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(x, per_movie, color=BLUE, lw=2)
    ax.set_yscale("log")
    ax.set_xlim(0, len(per_movie))
    ax.set_ylim(0.8, per_movie.max() * 1.5)
    ax.set_xlabel("Movies, sorted by number of ratings (descending)")
    ax.set_ylabel("Number of ratings (log scale)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: vn_int(v)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: vn_int(v) if v >= 1 else ""))
    ax.axvline(top10, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(top10 + 120, per_movie.max() / 2, f"Top 10% of movies ({vn_int(top10)} movies)\n"
            f"receive {vn(top10_share, 0)}% of all ratings", fontsize=9, color=INK, va="center")
    first_one = int(np.argmax(per_movie <= 1)) + 1
    ax.axvline(first_one, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(first_one + 120, 6, f"{vn_int(n_one)} movies ({vn(100 * n_one / len(per_movie))}%)\n"
            "have exactly 1 rating", fontsize=9, color=INK, va="center")
    fig.tight_layout()
    fig.savefig(OUT / f"{PREFIX}fig3_2_long_tail.png")
    plt.close(fig)

    # Figure 3.3: shared raters per movie pair (training set)
    C = co_counts(rm_train)
    vals = C[C > 0]
    del C
    bins = [("1", vals == 1), ("2", vals == 2), ("3", vals == 3), ("4", vals == 4),
            ("5–9", (vals >= 5) & (vals <= 9)), ("10–19", (vals >= 10) & (vals <= 19)), ("≥ 20", vals >= 20)]
    shares = [100 * b.mean() for _, b in bins]
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=200)
    ax.bar([b for b, _ in bins], shares, color=BLUE, width=0.72)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Number of users who rated both movies")
    ax.set_ylabel("Share of movie pairs (%)")
    for i, v in enumerate(shares):
        ax.annotate(f"{vn(v)}%", (i, v), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    ax.set_ylim(0, max(shares) * 1.15)
    fig.tight_layout()
    fig.savefig(OUT / f"{PREFIX}fig3_3_shared_raters.png")
    plt.close(fig)
    print("Drawn: extra_fig3_1_rating_distribution.png, extra_fig3_2_long_tail.png, extra_fig3_3_shared_raters.png")


# =====================================================================
# CHAPTER 5
# =====================================================================
def top_lists(rm, scores, k, min_ratings):
    """The first K movies of every user, filtered exactly as in evaluate.topn_metrics."""
    blocked = (rm.M > 0) | (rm.item_count < min_ratings)[None, :] | np.isnan(scores)
    s = np.where(blocked, -np.inf, scores)
    top = np.argpartition(-s, k, axis=1)[:, :k]
    valid = np.isfinite(np.take_along_axis(s, top, axis=1))
    return top, valid


def popularity_table(rm, ts):
    """Table 5.4: popularity of the movies that appear in the top-5 lists."""
    header("TABLE 5.4. POPULARITY OF MOVIES IN THE TOP-5 LISTS")
    K = config.EVAL_K
    plans = [
        ("E1 Cosine, first version", dict(measure="cosine", k=None, beta=0, rank_by="predicted", m=0)),
        ("Cosine, k=50, beta=10, ranked by predicted rating", dict(measure="cosine", k=50, beta=10, rank_by="predicted", m=config.MIN_ITEM_RATINGS)),
        ("E2 Cosine, ranked by summed similarity", dict(measure="cosine", k=50, beta=10, rank_by="sum", m=config.MIN_ITEM_RATINGS)),
        ("E3 Pearson", dict(measure="pearson", k=50, beta=10, rank_by="sum", m=config.MIN_ITEM_RATINGS)),
        ("E4 Jaccard", dict(measure="jaccard", k=50, beta=10, rank_by="sum", m=config.MIN_ITEM_RATINGS)),
    ]
    rows = []
    for name, p in plans:
        S = build_similarity(rm, p["measure"], p["k"], p["beta"], use_cache=p["k"] is not None, verbose=False)
        P = predict_all(rm, S, centered=config.centered_for(p["measure"]))
        scores = rank_scores(rm, S) if p["rank_by"] == "sum" else P
        top, valid = top_lists(rm, scores, K, p["m"])
        counts = rm.item_count[top][valid]
        prec = topn_metrics(scores, rm, ts, K, p["m"])["precision"]
        rows.append({
            "configuration": name,
            "median_ratings": float(np.median(counts)),
            "share_le_2_ratings_pct": round(100 * float((counts <= 2).mean()), 1),
            "distinct_movies": int(len(np.unique(top[valid]))),
            "precision_at_5": round(prec, 4),
        })
        r = rows[-1]
        print(f"{name:52s} median {vn(r['median_ratings'], 0):>4} ratings | <=2 ratings {vn(r['share_le_2_ratings_pct'])}% "
              f"| {vn_int(r['distinct_movies'])} distinct movies | P@5 {vn(prec, 4)}")
    print("Saved:", save_csv(pd.DataFrame(rows), "table5_4_popularity").name)


def examples_user1(rm_full, movies):
    """Table 5.5: first 5 recommendations for profile 1 with Cosine and Pearson."""
    header("TABLE 5.5. FIRST FIVE RECOMMENDATIONS FOR PROFILE 1")
    titles = movies.set_index("movieId")["title"]
    u = rm_full.user_index(1)
    print(f"Profile 1: {int(rm_full.M[u].sum())} movies rated, mean rating {vn(rm_full.user_mean[u], 2)}")
    rows = []
    for measure in ("cosine", "pearson"):
        S = build_similarity(rm_full, measure, verbose=False)
        recs = recommend_top_n(rm_full, S, 1, n=5, centered=config.centered_for(measure))
        print(f"\n{measure.capitalize()}:")
        for rank, r in enumerate(recs, 1):
            n_ratings = int(rm_full.item_count[rm_full.movie_indexer([r["movieId"]])[0]])
            rows.append({"measure": measure, "rank": rank, "movie": titles[r["movieId"]],
                         "n_ratings": n_ratings, "predicted": r["predicted"],
                         "because_liked": titles[r["because"]]})
            print(f"  {rank}. {titles[r['movieId']]} ({n_ratings} ratings) - because they liked {titles[r['because']]}")
    print("\nSaved:", save_csv(pd.DataFrame(rows), "table5_5_profile1").name)


def performance(ratings, rm_full, rm_train, ts):
    """Table 5.2: performance measured on the running machine."""
    header("TABLE 5.2. PERFORMANCE (measured on this machine)")
    rows = []

    # Build each similarity matrix on the full data, without the cache
    for measure in ("cosine", "pearson", "jaccard"):
        t0 = time.perf_counter()
        S = build_similarity(rm_full, measure, use_cache=False, verbose=False)
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        for uid in rm_full.user_ids:
            recommend_top_n(rm_full, S, uid, n=20, centered=config.centered_for(measure))
        t_rec = (time.perf_counter() - t0) * 1000 / len(rm_full.user_ids)
        buf = io.BytesIO()
        sparse.save_npz(buf, S)
        size_mb = buf.tell() / 1e6
        rows += [
            {"measurement": f"Build the {measure} similarity matrix (full data, no cache)", "result": round(t_build, 2), "unit": "s"},
            {"measurement": f"20 recommendations for one user ({measure}), mean over {len(rm_full.user_ids)} users", "result": round(t_rec, 2), "unit": "ms"},
            {"measurement": f"Size of the {measure} cache file (k = {config.K_NEIGHBORS})", "result": round(size_mb, 2), "unit": "MB"},
        ]
        print(f"{measure:8s} build S: {vn(t_build, 2)} s | recommend: {vn(t_rec, 2)} ms/user | cache file: {vn(size_mb, 2)} MB")

    # Evaluate one configuration (E2) from scratch on the full test set, with peak memory
    tracemalloc.start()
    t0 = time.perf_counter()
    S = build_similarity(rm_train, "cosine", use_cache=False, verbose=False)
    P = predict_all(rm_train, S)
    rating_metrics(P, rm_train, ts)
    topn_metrics(rank_scores(rm_train, S), rm_train, ts)
    t_eval = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rows += [
        {"measurement": "Evaluate configuration E2 on the full test set (from scratch)", "result": round(t_eval, 2), "unit": "s"},
        {"measurement": "Peak memory while evaluating E2 (allocated by Python/NumPy)", "result": round(peak / 1e6), "unit": "MB"},
    ]
    print(f"Evaluate E2 from scratch: {vn(t_eval, 2)} s | peak memory: {vn_int(peak / 1e6)} MB")
    print("Saved:", save_csv(pd.DataFrame(rows), "table5_2_performance").name)
    print("Note: the web start-up time and the evaluate.py run time are printed by those two programs.")


# =====================================================================
def main(no_perf: bool):
    t_all = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    print("Reading data...")
    ratings = load_ratings()
    movies = load_movies()
    train_df, test_df = split_train_test(ratings)
    rm_train = build_matrix(train_df, tag="train")
    rm_full = build_matrix(ratings, tag="full")
    ts = TestSet(rm_train, test_df)

    chapter3(ratings, rm_train)
    popularity_table(rm_train, ts)
    examples_user1(rm_full, movies)
    if not no_perf:
        performance(ratings, rm_full, rm_train, ts)

    print(f"\nDone in {vn(time.perf_counter() - t_all)} s. New files are in {OUT} (names start with '{PREFIX}').")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute the additional numbers for the report")
    parser.add_argument("--no-perf", action="store_true", help="skip the performance part (Table 5.2)")
    main(parser.parse_args().no_perf)
