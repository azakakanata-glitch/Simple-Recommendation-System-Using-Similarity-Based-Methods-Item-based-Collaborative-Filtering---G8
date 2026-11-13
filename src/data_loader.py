"""
data_loader.py
Owner: Nguyen Khanh Toan
Purpose: read the MovieLens data, split it into train/test and build the user-movie matrix.

Every later step (similarity, recommender, evaluate, app) receives a RatingMatrix
object instead of rebuilding the matrix, so the whole system shares one
userId/movieId indexing scheme (requirement NFR3).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


# ================= READING FILES =================
def load_ratings(path=config.RATINGS_FILE) -> pd.DataFrame:
    """Read ratings.csv: userId, movieId, rating, timestamp."""
    return pd.read_csv(path)


def load_movies(path=config.MOVIES_FILE) -> pd.DataFrame:
    """Read movies.csv and extract the release year from the title."""
    movies = pd.read_csv(path)
    year = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    movies["year"] = pd.to_numeric(year, errors="coerce").astype("Int64")
    movies["genres"] = movies["genres"].replace("(no genres listed)", "")
    return movies


def split_train_test(ratings: pd.DataFrame, test_size=config.TEST_SIZE,
                     seed=config.RANDOM_STATE):
    """Random split by rating (80/20 by default) with a fixed seed."""
    train_df, test_df = train_test_split(ratings, test_size=test_size, random_state=seed)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ================= MATRIX =================
@dataclass
class RatingMatrix:
    """
    R          : rating matrix (users x movies), 0 = not rated, float32
    M          : indicator matrix of the same shape, 1 = rated, float32
    user_ids   : userId of each row
    movie_ids  : movieId of each column
    user_mean, item_mean : mean rating of each user / each movie
    item_count : number of ratings of each movie
    tag        : label of the data set ("train" or "full"), used in cache file names
    """
    R: np.ndarray
    M: np.ndarray
    user_ids: np.ndarray
    movie_ids: np.ndarray
    user_mean: np.ndarray
    item_mean: np.ndarray
    item_count: np.ndarray
    global_mean: float
    n_ratings: int
    tag: str

    @property
    def shape(self):
        return self.R.shape

    @property
    def fingerprint(self) -> str:
        """Data fingerprint, added to cache file names so an outdated matrix is never reused."""
        return f"{self.tag}{self.n_ratings}"

    def user_index(self, user_id) -> int:
        """Row index of a userId; -1 if it does not exist."""
        pos = np.searchsorted(self.user_ids, user_id)
        if pos < len(self.user_ids) and self.user_ids[pos] == user_id:
            return int(pos)
        return -1

    def user_indexer(self, user_ids) -> np.ndarray:
        """Vectorised user_index for many userIds."""
        return pd.Index(self.user_ids).get_indexer(user_ids)

    def movie_indexer(self, movie_ids) -> np.ndarray:
        """Column indices for many movieIds; -1 if a movie is not in the matrix."""
        return pd.Index(self.movie_ids).get_indexer(movie_ids)


def build_matrix(ratings_df: pd.DataFrame, tag: str = "full") -> RatingMatrix:
    """
    Turn long-format data (one row = one rating) into a RatingMatrix.
    Uses np.searchsorted instead of pivot_table: faster, and keeps the id order.
    """
    user_ids = np.sort(ratings_df["userId"].unique())
    movie_ids = np.sort(ratings_df["movieId"].unique())
    rows = np.searchsorted(user_ids, ratings_df["userId"].to_numpy())
    cols = np.searchsorted(movie_ids, ratings_df["movieId"].to_numpy())

    R = np.zeros((len(user_ids), len(movie_ids)), dtype=np.float32)
    R[rows, cols] = ratings_df["rating"].to_numpy(dtype=np.float32)
    M = (R > 0).astype(np.float32)

    user_cnt = M.sum(axis=1)
    item_cnt = M.sum(axis=0)
    user_mean = R.sum(axis=1) / np.maximum(user_cnt, 1)
    item_mean = R.sum(axis=0) / np.maximum(item_cnt, 1)

    return RatingMatrix(
        R=R, M=M, user_ids=user_ids, movie_ids=movie_ids,
        user_mean=user_mean.astype(np.float32), item_mean=item_mean.astype(np.float32),
        item_count=item_cnt.astype(np.int32), global_mean=float(ratings_df["rating"].mean()),
        n_ratings=len(ratings_df), tag=tag,
    )


if __name__ == "__main__":
    ratings = load_ratings()
    rm = build_matrix(ratings)
    density = rm.n_ratings / (rm.shape[0] * rm.shape[1])
    print(f"User-movie matrix: {rm.shape[0]} x {rm.shape[1]}, density {density:.2%}")
    train_df, test_df = split_train_test(ratings)
    print(f"Train: {len(train_df)} ratings | Test: {len(test_df)} ratings")
