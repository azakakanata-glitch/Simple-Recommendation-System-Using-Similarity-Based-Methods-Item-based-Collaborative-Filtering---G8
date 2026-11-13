"""
evaluate.py
Owner: Nguyen Khanh Toan
Purpose: offline evaluation of the system following the design in Section 3.6:
         - 80/20 train/test split (random_state = 42); every matrix is built from train only;
         - RMSE, MAE, Coverage for rating prediction;
         - Precision@K, Recall@K for the recommendation lists;
         - baselines B1-B3, configurations E1-E5 and sensitivity runs for k, beta, m.
Results are written to the results/ folder, where Chapter 5 takes its numbers from.

How to run (from the project root folder):
    python evaluate.py            # run everything
    python evaluate.py --quick    # only B1-B3 and E1-E5
"""
import argparse
import time

import numpy as np
import pandas as pd

from src import config
from src.data_loader import build_matrix, load_ratings, split_train_test
from src.recommender import predict_all, rank_scores
from src.similarity import build_similarity


# ================= METRICS =================
class TestSet:
    """Pre-computes the row/column indices of the test set in the training matrix."""

    def __init__(self, rm, test_df, threshold=config.RELEVANT_THRESHOLD):
        self.ui = rm.user_indexer(test_df["userId"])
        self.ii = rm.movie_indexer(test_df["movieId"])
        self.actual = test_df["rating"].to_numpy(dtype=np.float32)
        self.n = len(test_df)
        # relevant movies per user: test rating >= threshold and the movie exists in train
        rel = (self.actual >= threshold) & (self.ui >= 0) & (self.ii >= 0)
        self.relevant = np.zeros(rm.shape, dtype=bool)
        self.relevant[self.ui[rel], self.ii[rel]] = True
        self.n_relevant = self.relevant.sum(axis=1)
        self.eval_users = self.n_relevant > 0


def rating_metrics(P: np.ndarray, rm, ts: TestSet) -> dict:
    """RMSE, MAE on the WHOLE test set; pairs that cannot be predicted fall back to the user mean."""
    pred = np.full(ts.n, np.nan, dtype=np.float32)
    ok = (ts.ui >= 0) & (ts.ii >= 0)
    pred[ok] = P[ts.ui[ok], ts.ii[ok]]
    coverage = float(np.mean(~np.isnan(pred)))
    fallback = np.where(ts.ui >= 0, rm.user_mean[np.maximum(ts.ui, 0)], rm.global_mean)
    full = np.where(np.isnan(pred), fallback, pred)
    err = full - ts.actual
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "coverage": coverage}


def topn_metrics(scores: np.ndarray, rm, ts: TestSet, k=config.EVAL_K,
                 min_ratings=config.MIN_ITEM_RATINGS) -> dict:
    """Precision@K, Recall@K: only movies the user has NOT watched in train and with enough ratings."""
    blocked = (rm.M > 0) | (rm.item_count < min_ratings)[None, :] | np.isnan(scores)
    s = np.where(blocked, -np.inf, scores)
    top = np.argpartition(-s, k, axis=1)[:, :k]
    valid = np.isfinite(np.take_along_axis(s, top, axis=1))
    hits = (ts.relevant[np.arange(rm.shape[0])[:, None], top] & valid).sum(axis=1)
    u = ts.eval_users
    return {"precision": float(np.mean(hits[u] / k)),
            "recall": float(np.mean(hits[u] / ts.n_relevant[u]))}


# ================= ONE CONFIGURATION =================
def evaluate(rm, ts: TestSet, name: str, measure="cosine", k=config.K_NEIGHBORS,
             beta=config.BETA, mode="item", rank_by="sum",
             min_ratings=config.MIN_ITEM_RATINGS, eval_k=config.EVAL_K) -> dict:
    """Evaluate one Item-based configuration (or User-based, for comparison)."""
    t0 = time.perf_counter()
    S = build_similarity(rm, measure, k, beta, mode, use_cache=k is not None, verbose=False)
    P = predict_all(rm, S, centered=config.centered_for(measure), mode=mode)
    scores = rank_scores(rm, S, mode) if rank_by == "sum" else P
    row = {"id": name, "measure": measure, "mode": mode, "k": k if k else "all",
           "beta": beta, "m": min_ratings, "rank_by": rank_by}
    row.update(rating_metrics(P, rm, ts))
    row.update(topn_metrics(scores, rm, ts, eval_k, min_ratings))
    row["time_s"] = round(time.perf_counter() - t0, 2)
    return row


def baselines(rm, ts: TestSet, eval_k=config.EVAL_K) -> list:
    """B1: global mean; B2: user mean; B3: most popular movies."""
    rows = []
    for name, value in [("B1", np.full(rm.shape, rm.global_mean, np.float32)),
                        ("B2", np.repeat(rm.user_mean[:, None], rm.shape[1], axis=1))]:
        r = {"id": name, "measure": "global mean" if name == "B1" else "user mean"}
        m = rating_metrics(value, rm, ts)
        m.pop("coverage")                  # baselines always predict, so coverage is meaningless
        r.update(m)
        rows.append(r)
    pop = np.repeat(rm.item_count[None, :].astype(np.float32), rm.shape[0], axis=0)
    r = {"id": "B3", "measure": "most popular"}
    r.update(topn_metrics(pop, rm, ts, eval_k, min_ratings=0))
    rows.append(r)
    return rows


# ================= RUN THE EXPERIMENTS =================
def run_experiments(quick=False):
    t_all = time.perf_counter()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading data and splitting train/test (80/20, random_state = 42)...")
    train_df, test_df = split_train_test(load_ratings())
    rm = build_matrix(train_df, tag="train")
    ts = TestSet(rm, test_df)
    print(f"  train {len(train_df)} ratings, test {len(test_df)} ratings, "
          f"matrix {rm.shape[0]} x {rm.shape[1]}, {int(ts.eval_users.sum())} users with relevant movies")

    print("\nBaselines and main configurations:")
    rows = baselines(rm, ts)
    plan = [
        ("E1", dict(measure="cosine", k=None, beta=0, rank_by="predicted", min_ratings=0)),
        ("E2", dict(measure="cosine")),
        ("E3", dict(measure="pearson")),
        ("E4", dict(measure="jaccard")),
        ("E5", dict(measure="cosine", mode="user")),
    ]
    for name, kw in plan:
        r = evaluate(rm, ts, name, **kw)
        rows.append(r)
        print(f"  {name}: RMSE {r['rmse']:.4f}  MAE {r['mae']:.4f}  Coverage {r['coverage']:.3f}  "
              f"P@{config.EVAL_K} {r['precision']:.4f}  R@{config.EVAL_K} {r['recall']:.4f}  ({r['time_s']}s)")
    main = pd.DataFrame(rows)
    main.to_csv(config.RESULTS_DIR / "results.csv", index=False)

    if not quick:
        sweeps = {
            "k": [dict(k=v) for v in (10, 20, 50, 100, None)],
            "beta": [dict(beta=v) for v in (0, 5, 10, 25)],
            "m": [dict(min_ratings=v) for v in (0, 5, 10, 20)],
        }
        for key, settings in sweeps.items():
            print(f"\nSensitivity to {key} (Cosine, other parameters at their defaults):")
            out = []
            for kw in settings:
                r = evaluate(rm, ts, f"S-{key}", measure="cosine", **kw)
                out.append(r)
                print(f"  {key}={kw[list(kw)[0]]}: RMSE {r['rmse']:.4f}  Coverage {r['coverage']:.3f}  "
                      f"P@{config.EVAL_K} {r['precision']:.4f}  R@{config.EVAL_K} {r['recall']:.4f}")
            pd.DataFrame(out).to_csv(config.RESULTS_DIR / f"sensitivity_{key}.csv", index=False)

        print("\nEffect of the ranking method (Cosine, k=50, beta=10):")
        r = evaluate(rm, ts, "X-predicted", measure="cosine", rank_by="predicted")
        print(f"  ranked by predicted rating: P@{config.EVAL_K} {r['precision']:.4f}  R@{config.EVAL_K} {r['recall']:.4f}")
        pd.DataFrame([r]).to_csv(config.RESULTS_DIR / "ranking_by_prediction.csv", index=False)

    print("\n" + "=" * 72)
    print("RESULTS (full test set)")
    print("=" * 72)
    cols = ["id", "measure", "mode", "k", "beta", "rmse", "mae", "coverage", "precision", "recall"]
    table = main.reindex(columns=cols).astype(object)
    table["mode"] = table["mode"].fillna("-")
    for c in ("k", "beta"):
        table[c] = table[c].map(lambda v: "-" if pd.isna(v) else (v if isinstance(v, str) else int(v)))
    for c in ("rmse", "mae", "coverage", "precision", "recall"):
        table[c] = table[c].map(lambda v: "-" if pd.isna(v) else f"{v:.4f}")
    print(table.to_string(index=False))
    print(f"\nResults written to {config.RESULTS_DIR}. Total time: {time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline evaluation of the movie recommender")
    parser.add_argument("--quick", action="store_true", help="only run B1-B3 and E1-E5")
    run_experiments(quick=parser.parse_args().quick)
