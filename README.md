# Simple Recommendation System Using Similarity-Based Methods (Item-based Collaborative Filtering)

Foundation Project, team G8. A movie recommender based on the similarity between movies, built on the MovieLens ml-latest-small dataset, with the CineMatch web application.

## Installation

Requires Python 3.10 or later.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS, Linux
pip install -r requirements.txt
```

Put `ratings.csv` and `movies.csv` from MovieLens ml-latest-small in the `data/` folder (download from https://grouplens.org/datasets/movielens/latest/). `tags.csv` is optional; it is only counted by `report_extras.py`.

## Running

Run every command from the project root folder.

| Task | Command |
| --- | --- |
| Offline evaluation, results written to `results/` | `python evaluate.py` (add `--quick` to run only B1–B3, E1–E5) |
| Chapter 5 charts | `python plot_results.py` |
| Extra figures and tables for the report | `python report_extras.py` (add `--no-perf` to skip the timings) |
| Web application | `python app.py`, then open http://127.0.0.1:5000 |
| Automated tests (16 cases) | `python -m unittest -v tests.test_system` |
| Save the test output to a file | `python -m unittest -v tests.test_system > results/test_output.txt 2>&1` |
| Try a single module | `python -m src.similarity`, `python -m src.recommender` |

The web application runs entirely on your machine and needs no internet connection.

## Structure

```
data/                 MovieLens data
src/config.py         all parameters (measure, k, β, m, N, K, thresholds, paths)
src/data_loader.py    reads the data, splits train/test, builds the RatingMatrix
src/similarity.py     Cosine, Pearson, Jaccard, β shrinkage, top-k, cache
src/recommender.py    rating prediction, ranking score, Top-N
src/data_stats.py     exploratory data statistics (Chapter 3) for the web page
evaluate.py           experiments B1–B3, E1–E5, sensitivity to k, β, m
plot_results.py       charts from results/*.csv
report_extras.py      extra figures and tables for Chapters 3 and 5
app.py                Flask APIs
templates/, static/   CineMatch interface
tests/                automated tests
cache/                computed similarity matrices (generated)
results/              result tables and charts (generated)
```

## Output files in `results/`

| File | Produced by |
| --- | --- |
| `results.csv`, `sensitivity_k.csv`, `sensitivity_beta.csv`, `sensitivity_m.csv`, `ranking_by_prediction.csv` | `evaluate.py` |
| `fig5_1_rmse.png`, `fig5_2_precision.png`, `fig5_3_sensitivity_k.png`, `fig5_4_sensitivity_beta.png` | `plot_results.py` |
| `extra_*.csv`, `extra_*.png` | `report_extras.py` |

## Main results (test set of 20,168 ratings)

| Configuration | RMSE | MAE | Coverage | Precision@5 | Recall@5 |
| --- | --- | --- | --- | --- | --- |
| B2 User mean | 0.9497 | 0.7386 | – | – | – |
| B3 Most popular | – | – | – | 0.1492 | 0.0676 |
| E1 Cosine, first version | 0.9225 | 0.7112 | 0.9595 | 0.0000 | 0.0000 |
| **E2 Cosine, k = 50, β = 10** | **0.9117** | **0.6828** | **0.9171** | **0.2405** | **0.1120** |
| E3 Pearson | 0.8952 | 0.6756 | 0.8878 | 0.0963 | 0.0558 |
| E4 Jaccard | 0.9019 | 0.6765 | 0.8948 | 0.2254 | 0.1142 |

## Team

- Nguyễn Khánh Toàn (leader): data_loader.py, evaluate.py, app.py
- Lê Nhật Minh: similarity.py, data_stats.py, web interface (templates, static)
- Bùi Minh Tân: recommender.py, plot_results.py, report_extras.py, tests

The MovieLens data belongs to GroupLens Research, University of Minnesota, and is used for study and research only.
