"""
tests/test_system.py
Purpose: automated tests for requirements FR1-FR6, NFR1, NFR3 (Chapter 5, Section 5.2).
  - TestCore    : runs on the real MovieLens data (TC01-TC07)
  - TestFormulas: checks the formulas on the 3 x 4 example matrix of Chapter 2 against hand calculations (TC11-TC14)
  - TestWeb     : tests the APIs with the Flask test client (TC08-TC10, TC15-TC16)
How to run (from the project root folder):  python -m unittest -v tests.test_system
"""
import time
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from src import config
from src.data_loader import build_matrix, load_ratings, split_train_test
from src.recommender import predict_all, rank_scores, recommend_top_n
from src.similarity import build_similarity, cosine, jaccard, pearson, shrink
from evaluate import TestSet, rating_metrics, topn_metrics

# Example matrix of Section 2.1 (3 users x 4 movies, "-" = not rated):
#            Movie 1 Movie 2 Movie 3 Movie 4
#   User 1     4       5       3       -
#   User 2     -       3       -       4
#   User 3     5       4       -       2
TOY = pd.DataFrame({"userId":  [1, 1, 1, 2, 2, 3, 3, 3],
                    "movieId": [1, 2, 3, 2, 4, 1, 2, 4],
                    "rating":  [4, 5, 3, 3, 4, 5, 4, 2]}, dtype=float).astype({"userId": int, "movieId": int})


class TestCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ratings = load_ratings()
        cls.rm = build_matrix(cls.ratings)

    # FR1 - data loading
    def test_fr1_matrix_shape_and_values(self):
        self.assertEqual(self.rm.shape, (610, 9724))
        self.assertEqual(int(self.rm.M.sum()), len(self.ratings))
        u, i = self.rm.user_index(1), self.rm.movie_indexer([1])[0]
        self.assertEqual(self.rm.R[u, i], 4.0)          # userId 1 rated Toy Story 4.0 stars

    def test_fr1_split_has_no_overlap(self):
        train, test = split_train_test(self.ratings)
        self.assertEqual(len(train) + len(test), len(self.ratings))
        key = lambda df: set(zip(df.userId, df.movieId))
        self.assertFalse(key(train) & key(test))

    # FR2 - similarity
    def test_fr2_similarity_properties(self):
        for fn in (cosine, pearson, jaccard):
            S = fn(self.rm)
            self.assertTrue(np.allclose(S, S.T, atol=1e-5), fn.__name__)
            self.assertTrue(np.all(np.diag(S) == 0), fn.__name__)
            self.assertLessEqual(float(S.max()), 1.0 + 1e-5)

    def test_fr2_top_k_and_shrink(self):
        S = build_similarity(self.rm, "cosine", k=50, beta=10, use_cache=False, verbose=False)
        self.assertTrue(sparse.issparse(S))
        self.assertLessEqual(int(np.diff(S.indptr).max()), 50)

    # FR3 - rating prediction
    def test_fr3_predictions_in_range(self):
        S = build_similarity(self.rm, "cosine", verbose=False)
        P = predict_all(self.rm, S)
        valid = P[~np.isnan(P)]
        self.assertTrue(np.all((valid >= config.RATING_MIN) & (valid <= config.RATING_MAX)))

    # FR4 - Top-N recommendation
    def test_fr4_top_n_unwatched_and_sorted(self):
        S = build_similarity(self.rm, "cosine", verbose=False)
        for user_id in (1, 414, 610):
            recs = recommend_top_n(self.rm, S, user_id, n=20)
            self.assertEqual(len(recs), 20)
            u = self.rm.user_index(user_id)
            cols = self.rm.movie_indexer([r["movieId"] for r in recs])
            self.assertTrue(np.all(self.rm.M[u, cols] == 0))          # unwatched movies only
            scores = [r["score"] for r in recs]
            self.assertEqual(scores, sorted(scores, reverse=True))   # descending
            self.assertTrue(np.all(self.rm.item_count[cols] >= config.MIN_ITEM_RATINGS))

    def test_fr4_unknown_user(self):
        S = build_similarity(self.rm, "cosine", verbose=False)
        with self.assertRaises(KeyError):
            recommend_top_n(self.rm, S, 99999)


class TestFormulas(unittest.TestCase):
    """Compare the code with hand calculations on the example matrix of Chapter 2."""

    @classmethod
    def setUpClass(cls):
        cls.rm = build_matrix(TOY, tag="toy")

    # TC11 - FR2: similarity values of Movie 1 - Movie 2
    def test_fr2_toy_similarity_values(self):
        # Cosine on full vectors, empty cells = 0 (Section 2.1.6): 40 / (sqrt(41) * sqrt(50)) = 0.883
        self.assertAlmostEqual(float(cosine(self.rm)[0, 1]), 40 / np.sqrt(41 * 50), places=4)
        # Jaccard: {1, 3} and {1, 2, 3} -> 2 / 3
        self.assertAlmostEqual(float(jaccard(self.rm)[0, 1]), 2 / 3, places=4)
        # Pearson (3.2): Movie 1 minus 4.5 -> (-0.5, 0, 0.5); Movie 2 minus 4 -> (1, -1, 0) => -0.5 / (sqrt(0.5) * sqrt(2)) = -0.5
        self.assertAlmostEqual(float(pearson(self.rm)[0, 1]), -0.5, places=4)

    # TC12 - FR2: shrinkage formula (3.4)
    def test_fr2_shrink_formula(self):
        S = np.ones((1, 3), dtype=np.float32)
        out = shrink(S, np.array([[1, 10, 20]], dtype=np.float32), 10)
        self.assertTrue(np.allclose(out, [[0.1, 1.0, 1.0]]))

    # TC13 - FR3: the matrix form (3.7) equals a loop over formulas (3.5) and (3.6)
    def test_fr3_matrix_form_equals_loop(self):
        R, M = self.rm.R, self.rm.M
        for measure, centered in (("cosine", False), ("pearson", True)):
            S = build_similarity(self.rm, measure, k=None, beta=0, use_cache=False, verbose=False)
            S = np.asarray(S)
            P = predict_all(self.rm, S, centered=centered)
            mu = self.rm.item_mean
            for u in range(R.shape[0]):
                for i in range(R.shape[1]):
                    js = [j for j in range(R.shape[1]) if M[u, j] > 0 and S[i, j] != 0]
                    den = sum(abs(S[i, j]) for j in js)
                    if den == 0:
                        self.assertTrue(np.isnan(P[u, i]))
                        continue
                    if centered:
                        exp = mu[i] + sum(S[i, j] * (R[u, j] - mu[j]) for j in js) / den
                    else:
                        exp = sum(S[i, j] * R[u, j] for j in js) / den
                    exp = min(max(exp, config.RATING_MIN), config.RATING_MAX)
                    self.assertAlmostEqual(float(P[u, i]), float(exp), places=4, msg=f"{measure} u={u} i={i}")

    # TC14 - FR5: RMSE, MAE, Coverage, Precision@K, Recall@K on a hand-made example
    def test_fr5_metrics_hand_example(self):
        test_df = pd.DataFrame({"userId": [1, 2], "movieId": [4, 1], "rating": [5.0, 3.0]})
        ts = TestSet(self.rm, test_df)
        P = np.full(self.rm.shape, 3.0, dtype=np.float32)
        m = rating_metrics(P, self.rm, ts)                  # errors 2 and 0
        self.assertAlmostEqual(m["rmse"], np.sqrt(2), places=4)
        self.assertAlmostEqual(m["mae"], 1.0, places=4)
        self.assertAlmostEqual(m["coverage"], 1.0, places=4)
        # User 1 has only Movie 4 left unwatched, and it is relevant (5 stars >= 4) -> 1 hit out of 1
        scores = np.ones(self.rm.shape, dtype=np.float32)
        t = topn_metrics(scores, self.rm, ts, k=1, min_ratings=0)
        self.assertAlmostEqual(t["precision"], 1.0, places=4)
        self.assertAlmostEqual(t["recall"], 1.0, places=4)


class TestWeb(unittest.TestCase):
    """FR6 - API tests with the Flask test client (no browser needed)."""

    @classmethod
    def setUpClass(cls):
        import app
        cls.client = app.app.test_client()

    def test_fr6_pages_and_apis(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        info = self.client.get("/api/info").get_json()
        self.assertEqual(info["n_users"], 610)
        profiles = self.client.get("/api/profiles").get_json()["profiles"]
        self.assertEqual(len(profiles), config.PROFILE_LIMIT)
        res = self.client.get("/api/recommend?user=414&n=12").get_json()
        self.assertEqual(len(res["items"]), 12)
        self.assertEqual(len(self.client.get("/api/watched?user=414&n=5").get_json()["items"]), 5)

    def test_fr6_input_validation(self):
        self.assertEqual(self.client.get("/api/recommend?user=99999").status_code, 404)
        self.assertEqual(self.client.get("/api/recommend?user=abc").status_code, 400)
        self.assertEqual(self.client.get("/api/recommend?user=1&n=1").status_code, 400)
        self.assertEqual(self.client.get("/api/recommend?user=1&n=21").status_code, 400)
        self.assertEqual(self.client.get("/api/recommend?user=1&measure=abc").status_code, 400)

    # TC15 - FR6: the two data APIs of the report (Chapters 3 and 5)
    def test_fr6_stats_and_results_apis(self):
        stats = self.client.get("/api/stats")
        self.assertEqual(stats.status_code, 200)
        self.assertEqual(stats.get_json()["overview"]["n_ratings"], 100836)
        res = self.client.get("/api/results")
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        if body["available"]:                       # evaluate.py has been run
            self.assertIn("E2", [r["id"] for r in body["rows"]])

    # TC16 - NFR3: the web app and the evaluation share one algorithm
    def test_nfr3_web_matches_evaluation_code(self):
        import app
        rm, S = app.RM, app.similarity_for("cosine")
        scores = rank_scores(rm, S)                 # the function evaluate.py uses
        P = predict_all(rm, S)
        u = rm.user_index(414)
        items = self.client.get("/api/recommend?user=414&n=10").get_json()["items"]
        for it in items:
            i = rm.movie_indexer([it["movieId"]])[0]
            self.assertAlmostEqual(it["predicted"], round(float(P[u, i]), 1), delta=0.051)
        top = [it["movieId"] for it in items]
        cols = rm.movie_indexer(top)
        self.assertEqual(list(scores[u, cols]), sorted(scores[u, cols], reverse=True))

    def test_nfr1_response_time(self):
        start = time.perf_counter()
        for user_id in range(1, 51):
            self.client.get(f"/api/recommend?user={user_id}&n=20")
        avg_ms = (time.perf_counter() - start) * 1000 / 50
        self.assertLess(avg_ms, 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
