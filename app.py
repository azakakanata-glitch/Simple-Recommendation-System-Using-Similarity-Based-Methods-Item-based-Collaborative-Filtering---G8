"""
app.py
Owner: Nguyen Khanh Toan
Purpose: the CineMatch web application (Flask) that demonstrates the recommender.

At start-up it builds the matrices from ALL the data (no train/test split, since the
goal is recommendation, not evaluation) and loads the similarity matrix from the cache.
Each request only computes on data already in memory and calls no external service (NFR1, NFR5).

API:
    GET /api/info                          data size and configuration
    GET /api/profiles?limit=60&q=<id>      list of profiles to choose from
    GET /api/recommend?user=<id>&n=<2..20>&measure=<cosine|pearson|jaccard>
    GET /api/watched?user=<id>&n=<count>
    GET /api/stats                         exploratory data statistics (Chapter 3)
    GET /api/results                       evaluation results (Chapter 5), read from results/results.csv

How to run:  python app.py   then open http://127.0.0.1:5000
"""
import re
import time

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src import config
from src.data_loader import build_matrix, load_movies, load_ratings
from src.data_stats import load_stats
from src.recommender import recommend_top_n, user_top_rated
from src.similarity import MEASURES, build_similarity

app = Flask(__name__)
app.json.ensure_ascii = False


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status, self.message = status, message


@app.errorhandler(ApiError)
def handle_api_error(err):
    return jsonify({"error": err.message}), err.status


# ================= LOAD THE DATA ONCE =================
def clean_title(title: str) -> str:
    """'Terminator, The (1984)' -> 'The Terminator' (the year is returned separately)."""
    title = re.sub(r"\s*\(\d{4}\)\s*$", "", title)
    main, sep, rest = title.partition(" (")
    match = re.match(r"^(.*), (The|A|An|Les|La|Le|L'|Il|El|Das|Der|Die)$", main)
    if match:
        main = f"{match.group(2)} {match.group(1)}".replace("L' ", "L'")
    return main + (sep + rest if sep else "")


print("Loading data...")
t0 = time.perf_counter()
RM = build_matrix(load_ratings(), tag="full")
MOVIES = load_movies().set_index("movieId")
SIMS = {config.MEASURE: build_similarity(RM, config.MEASURE)}

# Profile summary: movies rated, mean rating, three most-watched genres
_user_counts = RM.M.sum(axis=1).astype(int)
_genre_names = sorted({g for gs in MOVIES["genres"] for g in gs.split("|") if g})
_genre_matrix = np.zeros((RM.shape[1], len(_genre_names)), dtype=np.float32)
for col, mid in enumerate(RM.movie_ids):
    for g in MOVIES.at[mid, "genres"].split("|"):
        if g:
            _genre_matrix[col, _genre_names.index(g)] = 1
_user_genres = RM.M @ _genre_matrix
PROFILES = []
for u, uid in enumerate(RM.user_ids):
    top = np.argsort(-_user_genres[u])[:3]
    PROFILES.append({"userId": int(uid), "n_ratings": int(_user_counts[u]),
                     "mean_rating": round(float(RM.user_mean[u]), 2),
                     "top_genres": [_genre_names[g] for g in top if _user_genres[u, g] > 0]})
PROFILES.sort(key=lambda p: -p["n_ratings"])
STATS = load_stats()          # Chapter 3: computed once, then read from cache/data_stats.json
print(f"Ready after {time.perf_counter() - t0:.1f}s: {RM.shape[0]} profiles, {RM.shape[1]} movies.")


def similarity_for(measure: str):
    """Similarity matrix of a measure; non-default measures are computed the first time they are needed."""
    if measure not in MEASURES:
        raise ApiError(400, f"The measure must be one of: {', '.join(MEASURES)}.")
    if measure not in SIMS:
        SIMS[measure] = build_similarity(RM, measure)
    return SIMS[measure]


def movie_info(movie_id: int) -> dict:
    row = MOVIES.loc[movie_id]
    return {"movieId": int(movie_id), "title": clean_title(row["title"]),
            "year": None if pd.isna(row["year"]) else int(row["year"]),
            "genres": [g for g in row["genres"].split("|") if g]}


def read_user() -> int:
    raw = request.args.get("user", "")
    if not raw.isdigit():
        raise ApiError(400, "The profile ID is missing or is not a number.")
    user_id = int(raw)
    if RM.user_index(user_id) < 0:
        raise ApiError(404, f"There is no profile with ID {user_id}.")
    return user_id


def read_int(name: str, default: int, low: int, high: int) -> int:
    raw = request.args.get(name)
    if raw is None:
        return default
    if not raw.isdigit() or not low <= int(raw) <= high:
        raise ApiError(400, f"Parameter {name} must be an integer from {low} to {high}.")
    return int(raw)


# ================= PAGE AND APIS =================
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/info")
def api_info():
    return jsonify({"n_users": RM.shape[0], "n_movies": RM.shape[1], "n_ratings": RM.n_ratings,
                    "measure": config.MEASURE, "measures": list(MEASURES),
                    "k": config.K_NEIGHBORS, "beta": config.BETA,
                    "n_min": config.TOP_N_MIN, "n_max": config.TOP_N_MAX,
                    "n_default": config.TOP_N_DEFAULT,
                    "min_item_ratings": config.MIN_ITEM_RATINGS, "eval_k": config.EVAL_K,
                    "relevant_threshold": config.RELEVANT_THRESHOLD,
                    "test_size": config.TEST_SIZE, "random_state": config.RANDOM_STATE})


@app.get("/api/stats")
def api_stats():
    return jsonify(STATS)


@app.get("/api/results")
def api_results():
    """The result table of evaluate.py; empty if the evaluation has not been run."""
    path = config.RESULTS_DIR / "results.csv"
    if not path.exists():
        return jsonify({"available": False, "rows": []})
    df = pd.read_csv(path)
    rows = []
    for r in df.to_dict(orient="records"):
        rows.append({k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()})
    return jsonify({"available": True, "rows": rows})


@app.get("/api/profiles")
def api_profiles():
    limit = read_int("limit", config.PROFILE_LIMIT, 1, len(PROFILES))
    query = request.args.get("q", "").strip()
    items = [p for p in PROFILES if str(p["userId"]).startswith(query)] if query else PROFILES
    return jsonify({"total": len(items), "profiles": items[:limit]})


@app.get("/api/recommend")
def api_recommend():
    user_id = read_user()
    n = read_int("n", config.TOP_N_DEFAULT, config.TOP_N_MIN, config.TOP_N_MAX)
    measure = request.args.get("measure", config.MEASURE)
    S = similarity_for(measure)
    t0 = time.perf_counter()
    recs = recommend_top_n(RM, S, user_id, n=n, centered=config.centered_for(measure))
    items = []
    for r in recs:
        info = movie_info(r["movieId"])
        info.update(predicted=r["predicted"],
                    because={"movieId": r["because"], "title": clean_title(MOVIES.at[r["because"], "title"])})
        items.append(info)
    return jsonify({"userId": user_id, "measure": measure, "n": n, "items": items,
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1)})


@app.get("/api/watched")
def api_watched():
    user_id = read_user()
    n = read_int("n", config.WATCHED_DEFAULT, 1, 50)
    items = []
    for r in user_top_rated(RM, user_id, n):
        info = movie_info(r["movieId"])
        info["rating"] = r["rating"]
        items.append(info)
    return jsonify({"userId": user_id, "items": items})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
