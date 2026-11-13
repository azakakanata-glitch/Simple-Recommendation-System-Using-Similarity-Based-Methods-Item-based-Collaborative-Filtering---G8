"""
recommender.py
Owner: Bui Minh Tan
Purpose: use the similarity matrix (from similarity.py) to
         - predict a rating for every user-movie pair (formulas 3.5 - 3.7),
         - compute the Top-N ranking score (formula 3.8),
         - build the recommendation list for one user (Algorithm 3.2).

The two roles are kept apart: the PREDICTED RATING is used for RMSE and shown to users;
the RANKING SCORE (summed similarity) is used to order the recommendations.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from src import config
from src.data_loader import RatingMatrix


# ================= HELPERS =================
def _abs(S):
    return abs(S) if sparse.issparse(S) else np.abs(S)


def _times(S, X) -> np.ndarray:
    """S @ X, always returned as numpy.ndarray (S may be sparse)."""
    out = S @ X
    return np.asarray(out.toarray() if sparse.issparse(out) else out, dtype=np.float32)


def _finish(num, den, base=None) -> np.ndarray:
    """Divide numerator by denominator, NaN when the denominator is 0 (not enough data), add back the mean, clip to [0.5, 5]."""
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.where(den > 1e-9, num / den, np.nan)
    if base is not None:
        P = P + base
    return np.clip(P, config.RATING_MIN, config.RATING_MAX).astype(np.float32)


# ================= PREDICT EVERYTHING =================
def predict_all(rm: RatingMatrix, S, centered: bool = False, mode: str = "item") -> np.ndarray:
    """
    Prediction matrix P (users x movies), NaN where there is not enough data.
      item, plain        (3.5):  P = (R S^T) / (M |S|^T)
      item, mean-centred (3.6):  P = mu_i + ((R - mu) S^T) / (M |S|^T)
      user (comparison)        :  P = (S R) / (|S| M), and likewise when centred
    """
    R, M = rm.R, rm.M
    if mode == "item":
        X = np.where(M > 0, R - rm.item_mean[None, :], 0.0).astype(np.float32) if centered else R
        num = _times(S, X.T).T
        den = _times(_abs(S), M.T).T
        return _finish(num, den, rm.item_mean[None, :] if centered else None)
    X = np.where(M > 0, R - rm.user_mean[:, None], 0.0).astype(np.float32) if centered else R
    num = _times(S, X)
    den = _times(_abs(S), M)
    return _finish(num, den, rm.user_mean[:, None] if centered else None)


def rank_scores(rm: RatingMatrix, S, mode: str = "item") -> np.ndarray:
    """Formula (3.8): score_ui = sum of s_ij over the movies j rated by u, i.e. Score = M S^T."""
    if mode == "item":
        return _times(S, rm.M.T).T
    return _times(S, rm.M)


# ================= RECOMMENDATIONS FOR ONE USER =================
def _user_vectors(rm: RatingMatrix, S, u: int, centered: bool):
    """Ranking scores and predicted ratings for ONE user (item-based), fast."""
    m_u, r_u = rm.M[u], rm.R[u]
    score = _times(S, m_u[:, None]).ravel()
    x = np.where(m_u > 0, r_u - rm.item_mean, 0.0).astype(np.float32) if centered else r_u
    num = _times(S, x[:, None]).ravel()
    den = _times(_abs(S), m_u[:, None]).ravel()
    pred = _finish(num, den, rm.item_mean if centered else None)
    return score, pred


def recommend_top_n(rm: RatingMatrix, S, user_id, n: int = config.TOP_N_DEFAULT,
                    min_ratings: int = config.MIN_ITEM_RATINGS,
                    centered: bool = False) -> list:
    """
    Algorithm 3.2. Returns at most n movies that user_id has NOT watched, ordered by ranking score.
    Each item: dict(movieId, score, predicted, because) - because is the movieId of the
    watched movie that contributes most to the recommendation (used to explain it).
    """
    u = rm.user_index(user_id)
    if u < 0:
        raise KeyError(f"userId {user_id} does not exist.")
    score, pred = _user_vectors(rm, S, u, centered)

    blocked = (rm.M[u] > 0) | (rm.item_count < min_ratings) | ~(score > 0) | np.isnan(pred)
    score = np.where(blocked, -np.inf, score)
    n = min(n, int(np.isfinite(score).sum()))
    if n == 0:
        return []
    top = np.argpartition(-score, n - 1)[:n]
    top = top[np.argsort(-score[top])]

    watched = np.flatnonzero(rm.M[u] > 0)
    result = []
    for i in top:
        row = S[i].toarray().ravel() if sparse.issparse(S) else np.asarray(S[i]).ravel()
        contrib = row[watched] * rm.R[u, watched]            # contribution of each watched movie
        j = watched[int(np.argmax(contrib))]
        result.append({
            "movieId": int(rm.movie_ids[i]),
            "score": round(float(score[i]), 4),
            "predicted": round(float(pred[i]), 2),
            "because": int(rm.movie_ids[j]),
        })
    return result


def recommend_with_titles(rm: RatingMatrix, S, user_id, movies: pd.DataFrame,
                          n: int = config.TOP_N_DEFAULT, **kwargs) -> pd.DataFrame:
    """Like recommend_top_n but with titles and genres - for printing or demos."""
    recs = recommend_top_n(rm, S, user_id, n=n, **kwargs)
    if not recs:
        return pd.DataFrame(columns=["title", "genres", "predicted", "because_title"])
    df = pd.DataFrame(recs)
    info = movies.set_index("movieId")
    df["title"] = df["movieId"].map(info["title"])
    df["genres"] = df["movieId"].map(info["genres"])
    df["because_title"] = df["because"].map(info["title"])
    return df[["title", "genres", "predicted", "because_title"]]


def user_top_rated(rm: RatingMatrix, user_id, n: int = config.WATCHED_DEFAULT) -> list:
    """The movies user_id rated highest, used for the 'Rated highly' tab."""
    u = rm.user_index(user_id)
    if u < 0:
        raise KeyError(f"userId {user_id} does not exist.")
    idx = np.flatnonzero(rm.M[u] > 0)
    # sort by rating, descending; on ties the more popular movie comes first
    order = np.lexsort((-rm.item_count[idx], -rm.R[u, idx]))[:n]
    return [{"movieId": int(rm.movie_ids[idx[o]]), "rating": float(rm.R[u, idx[o]])} for o in order]


if __name__ == "__main__":
    from src.data_loader import build_matrix, load_movies, load_ratings
    from src.similarity import build_similarity

    rm = build_matrix(load_ratings())
    S = build_similarity(rm, "cosine")
    movies = load_movies()
    print("\nTop-10 recommendations for userId = 1:")
    print(recommend_with_titles(rm, S, 1, movies, n=10).to_string(index=False))
