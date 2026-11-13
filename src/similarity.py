"""
similarity.py
Owner: Le Nhat Minh
Purpose: compute the movie-movie similarity matrix (Item-based) with three measures,
         Cosine, Pearson and Jaccard; shrink by the number of shared raters (beta);
         keep the k strongest neighbours; read/write the cache.

Every function computes ALL pairs at once with matrix multiplication
(formulas 3.1 - 3.4 in the report) instead of looping over movie pairs.
mode="user" computes user-user similarity and is only used for comparison (E5).
"""
import time

import numpy as np
from scipy import sparse

from src import config
from src.data_loader import RatingMatrix

MEASURES = ("cosine", "pearson", "jaccard")


# ================= HELPERS =================
def _vectors(rm: RatingMatrix, mode: str):
    """Return (X, X_mask): each ROW is one object to compare (a movie or a user)."""
    if mode == "item":
        return rm.R.T, rm.M.T
    if mode == "user":
        return rm.R, rm.M
    raise ValueError(f"invalid mode: {mode}")


def _cosine_rows(X: np.ndarray) -> np.ndarray:
    """Cosine between every pair of rows of X. An all-zero row gives similarity 0."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = (X / norms).astype(np.float32)
    S = Xn @ Xn.T
    np.fill_diagonal(S, 0.0)                     # an object is not its own neighbour
    return S


# ================= THE THREE MEASURES =================
def cosine(rm: RatingMatrix, mode: str = "item") -> np.ndarray:
    """Formula (3.1): unrated cells count as 0."""
    X, _ = _vectors(rm, mode)
    return _cosine_rows(X)


def pearson(rm: RatingMatrix, mode: str = "item") -> np.ndarray:
    """
    Formula (3.2): subtract each object's mean (on rated cells only), then take
    the cosine. Unlike pandas.corr() it works on the full vectors, so it does not
    produce a coefficient of 1.0 when two movies share only two raters.
    """
    X, mask = _vectors(rm, mode)
    mean = rm.item_mean if mode == "item" else rm.user_mean
    Z = np.where(mask > 0, X - mean[:, None], 0.0).astype(np.float32)
    return _cosine_rows(Z)


def co_counts(rm: RatingMatrix, mode: str = "item") -> np.ndarray:
    """n_ij = number of users who rated both movies i and j (N = M^T M)."""
    _, mask = _vectors(rm, mode)
    C = mask @ mask.T
    np.fill_diagonal(C, 0.0)
    return C


def jaccard(rm: RatingMatrix, mode: str = "item", counts: np.ndarray = None) -> np.ndarray:
    """Formula (3.3): |U_i intersect U_j| / |U_i union U_j|, ignoring rating values."""
    C = co_counts(rm, mode) if counts is None else counts
    _, mask = _vectors(rm, mode)
    size = mask.sum(axis=1)
    union = size[:, None] + size[None, :] - C
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(union > 0, C / union, 0.0).astype(np.float32)
    np.fill_diagonal(S, 0.0)
    return S


# ================= POST-PROCESSING =================
def shrink(S: np.ndarray, counts: np.ndarray, beta) -> np.ndarray:
    """Formula (3.4): s'_ij = s_ij * min(n_ij, beta) / beta. Modifies S in place to save memory."""
    if not beta:
        return S
    S *= np.minimum(counts, beta) / np.float32(beta)
    return S


def keep_top_k(S: np.ndarray, k):
    """
    Keep only the k largest values of each row (k neighbours); the rest become 0.
    Returns a sparse matrix (scipy.sparse.csr_matrix) when k is set, to save space.
    """
    n = S.shape[0]
    if k is None or k >= n - 1:
        return S
    cols = np.argpartition(-S, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    vals = S[np.arange(n)[:, None], cols].ravel()
    keep = vals != 0
    return sparse.csr_matrix((vals[keep], (rows[keep], cols.ravel()[keep])),
                             shape=S.shape, dtype=np.float32)


# ================= MAIN FUNCTION =================
def cache_path(rm: RatingMatrix, measure: str, k, beta, mode: str):
    k_txt = "all" if k is None else str(k)
    name = f"sim_{measure}_{mode}_k{k_txt}_b{beta or 0}_{rm.fingerprint}"
    return config.CACHE_DIR / (name + (".npy" if k is None else ".npz"))


def build_similarity(rm: RatingMatrix, measure: str = config.MEASURE,
                     k=config.K_NEIGHBORS, beta=config.BETA, mode: str = "item",
                     use_cache: bool = True, verbose: bool = True):
    """
    Algorithm 3.1: compute S -> shrink -> keep top-k -> write the cache.
    Result: numpy.ndarray (when k=None) or scipy.sparse.csr_matrix.
    """
    if measure not in MEASURES:
        raise ValueError(f"The measure must be one of {MEASURES}")
    path = cache_path(rm, measure, k, beta, mode)
    if use_cache and path.exists():
        if verbose:
            print(f"  reading cache {path.name}")
        return np.load(path) if path.suffix == ".npy" else sparse.load_npz(path)

    t0 = time.perf_counter()
    counts = co_counts(rm, mode) if (beta or measure == "jaccard") else None
    if measure == "cosine":
        S = cosine(rm, mode)
    elif measure == "pearson":
        S = pearson(rm, mode)
    else:
        S = jaccard(rm, mode, counts)
    if beta:
        S = shrink(S, counts, beta)
    del counts
    S = keep_top_k(S, k)

    if use_cache:
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if sparse.issparse(S):
            sparse.save_npz(path, S)
        else:
            np.save(path, S)
    if verbose:
        print(f"  computed {measure} ({mode}, k={k}, beta={beta}) in {time.perf_counter() - t0:.1f}s")
    return S


def top_similar_items(rm: RatingMatrix, S, movie_id: int, n: int = 5):
    """Helper for quick checks: the n movies most similar to movie_id, as (movieId, similarity)."""
    j = rm.movie_indexer([movie_id])[0]
    if j < 0:
        raise ValueError(f"movieId {movie_id} does not exist in the data.")
    row = S[j].toarray().ravel() if sparse.issparse(S) else np.asarray(S[j]).ravel()
    order = np.argsort(-row)[:n]
    return [(int(rm.movie_ids[i]), float(row[i])) for i in order if row[i] > 0]


if __name__ == "__main__":
    from src.data_loader import build_matrix, load_movies, load_ratings

    rm = build_matrix(load_ratings())
    movies = load_movies().set_index("movieId")["title"]
    for m in MEASURES:
        S = build_similarity(rm, m)
        print(f"\nThe 5 movies most similar to 'Toy Story (1995)' by {m}:")
        for mid, s in top_similar_items(rm, S, 1, 5):
            print(f"   {s:.3f}  {movies[mid]}")
