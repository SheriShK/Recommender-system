"""
Глубокий подбор гиперпараметров BM25, ItemKNN (Cosine), TF-IDF на RetailRocket.
ALS/BPR не используются. Временная валидация (скользящие 60/20/20 на train-событиях).
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import pickle
import sys
import warnings
from collections import defaultdict
from itertools import product
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
try:
    from implicit.utils import ParameterWarning as ImplicitParameterWarning

    warnings.filterwarnings("ignore", category=ImplicitParameterWarning)
except ImportError:
    ImplicitParameterWarning = None  # type: ignore[misc, assignment]
try:
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

from implicit.nearest_neighbours import BM25Recommender, CosineRecommender, TFIDFRecommender

from train_models import (
    csr_int32_for_implicit,
    get_popular_items,
    load_data,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# -----------------------------------------------------------------------------
DATA_DIR = "data"
PLOTS_DIR = os.path.join(DATA_DIR, "tuning_plots")
NUM_THREADS_DEFAULT = 8

K_GRID = [10, 20, 30, 50, 75, 100, 150, 200]
BM25_B_GRID = [0.5, 0.75, 1.0]
BM25_K1_GRID = [0.5, 1.0, 1.5, 2.0]

ITEMKNN_APPROX_OPTS = [True, False]
ITEMKNN_THREAD_OPTS = [4, 8]

COMBINED_W_P = 0.2
COMBINED_W_R = 0.5
COMBINED_W_N = 0.3

EARLY_STOP_PATIENCE = 3
EARLY_STOP_MIN_DELTA = 0.001

N_CV_FOLDS = 3

os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(DATA_DIR, "hyperparameter_tuning_log.txt"), encoding="utf-8"),
    ],
)
log = logging.getLogger("tune")


def combined_score(p10: float, r10: float, n10: float) -> float:
    return COMBINED_W_R * r10 + COMBINED_W_N * n10 + COMBINED_W_P * p10


def _cosine_recommender_factory(K: int, num_threads: int, approximate_similarities: bool) -> CosineRecommender:
    params = inspect.signature(CosineRecommender.__init__).parameters
    kwargs: dict[str, Any] = {"K": K, "num_threads": num_threads}
    if "approximate_similarities" in params:
        kwargs["approximate_similarities"] = approximate_similarities
    return CosineRecommender(**kwargs)


def truncate_similarity_top_k(sim: csr_matrix, k: int) -> csr_matrix:
    """
    Оставить в каждой строке не более k ненулей с наибольшими весами.
    Всегда выбираем top-min(k, nnz_row) по значению (а не «всю строку», если nnz ≤ k),
    чтобы поведение было согласовано для любого K (иначе K=10 и K=20 могли
    получать разный набор рёбер при одной и той же полной similarity).
    """
    sim = sim.tocsr()
    if k <= 0 or sim.nnz == 0:
        return sim
    rows, cols, data = [], [], []
    for i in range(sim.shape[0]):
        lo, hi = sim.indptr[i], sim.indptr[i + 1]
        idx = sim.indices[lo:hi]
        val = sim.data[lo:hi]
        if len(idx) == 0:
            continue
        take = min(k, len(idx))
        # устойчивый выбор top-take (до 200 соседей — argsort приемлем)
        sel = np.argsort(-val, kind="mergesort")[:take]
        rows.extend([i] * take)
        cols.extend(idx[sel].tolist())
        data.extend(val[sel].tolist())
    out = csr_matrix((data, (rows, cols)), shape=sim.shape, dtype=np.float64)
    out.sort_indices()
    out.sum_duplicates()
    out.eliminate_zeros()
    return out


def attach_scorer(model: CosineRecommender | BM25Recommender | TFIDFRecommender) -> None:
    from implicit._nearest_neighbours import NearestNeighboursScorer

    if model.similarity is not None:
        model.scorer = NearestNeighboursScorer(model.similarity)


class CachedCosineRecommender(CosineRecommender):
    """Документирует идею кэша: полный fit с max K, затем усечение similarity по строкам."""

    pass


# -----------------------------------------------------------------------------
def train_test_split_temporal(data: dict, test_ratio: float = 0.2):
    """
    Как train_models.train_test_split: последние ~test_ratio событий пользователя → test.
    Дополнительно: список train-событий (для time-based CV) с timestamp.
    """
    interactions_df = data["interactions"].copy()
    interactions_df["user_id"] = interactions_df["user_id"].astype(np.int64)
    interactions_df["item_id"] = interactions_df["item_id"].astype(np.int64)
    interactions_df["relevance"] = interactions_df["relevance"].astype(np.float64)

    if "timestamp" not in interactions_df.columns:
        raise ValueError("В interactions.csv нужна колонка timestamp (Unix ms).")

    interactions_df["timestamp"] = pd.to_numeric(interactions_df["timestamp"], errors="coerce")
    interactions_df = interactions_df.dropna(subset=["timestamp"])
    interactions_df["timestamp"] = interactions_df["timestamp"].astype(np.int64)

    user_to_idx = {int(k): int(v) for k, v in data["user_to_idx"].items()}
    item_to_idx = {int(k): int(v) for k, v in data["item_to_idx"].items()}

    interactions_df = interactions_df[
        interactions_df["user_id"].isin(user_to_idx.keys()) & interactions_df["item_id"].isin(item_to_idx.keys())
    ]

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)

    tr_rows, tr_cols, tr_conf, tr_bin = [], [], [], []
    pur_rows, pur_cols, pur_data = [], [], []
    test_dict: dict[int, list[int]] = {}
    train_events: list[dict[str, Any]] = []

    for user_id, group in interactions_df.groupby("user_id"):
        u_idx = user_to_idx.get(int(user_id))
        if u_idx is None:
            continue

        g = group.sort_values("timestamp", kind="mergesort")
        idxs = g.index.to_numpy()
        n = len(idxs)

        if n < 2:
            for _, row in g.iterrows():
                i_idx = item_to_idx[int(row["item_id"])]
                tr_rows.append(u_idx)
                tr_cols.append(i_idx)
                tr_conf.append(float(row["relevance"]))
                tr_bin.append(1.0)
                if float(row["relevance"]) >= 5 or str(row.get("action_type", "")).lower() == "transaction":
                    pur_rows.append(u_idx)
                    pur_cols.append(i_idx)
                    pur_data.append(1.0)
                train_events.append(
                    {
                        "user_idx": u_idx,
                        "item_idx": i_idx,
                        "timestamp": int(row["timestamp"]),
                        "relevance": float(row["relevance"]),
                    }
                )
            continue

        n_test = max(1, min(int(np.ceil(n * test_ratio)), n - 1))
        test_idx_set = set(idxs[-n_test:])
        test_items_set = set()

        for idx, row in g.iterrows():
            i_idx = item_to_idx[int(row["item_id"])]
            rel = float(row["relevance"])
            ts = int(row["timestamp"])
            if idx in test_idx_set:
                test_items_set.add(i_idx)
            else:
                tr_rows.append(u_idx)
                tr_cols.append(i_idx)
                tr_conf.append(rel)
                tr_bin.append(1.0)
                train_events.append(
                    {
                        "user_idx": u_idx,
                        "item_idx": i_idx,
                        "timestamp": ts,
                        "relevance": rel,
                    }
                )
                if rel >= 5 or str(row.get("action_type", "")).lower() == "transaction":
                    pur_rows.append(u_idx)
                    pur_cols.append(i_idx)
                    pur_data.append(1.0)

        if test_items_set:
            test_dict[u_idx] = list(test_items_set)

    def _aggregate(rows, cols, conf, binary=False):
        if not rows:
            return csr_matrix((n_users, n_items))
        df_agg = pd.DataFrame({"r": rows, "c": cols, "v": conf})
        if binary:
            g = df_agg.groupby(["r", "c"], sort=False).size()
            rows_m = g.index.get_level_values(0).astype(np.int32).tolist()
            cols_m = g.index.get_level_values(1).astype(np.int32).tolist()
            data_m = [1.0] * len(rows_m)
        else:
            g = df_agg.groupby(["r", "c"], sort=False)["v"].sum()
            rows_m = g.index.get_level_values(0).astype(np.int32).tolist()
            cols_m = g.index.get_level_values(1).astype(np.int32).tolist()
            data_m = g.values.astype(np.float64).tolist()
        return csr_matrix((data_m, (rows_m, cols_m)), shape=(n_users, n_items))

    train_confidence = _aggregate(tr_rows, tr_cols, tr_conf, binary=False)
    train_binary = _aggregate(tr_rows, tr_cols, tr_bin, binary=True)
    train_purchase_binary = _aggregate(pur_rows, pur_cols, pur_data, binary=True) if pur_rows else csr_matrix((n_users, n_items))

    test_users = [u for u in test_dict if train_confidence.indptr[u + 1] > train_confidence.indptr[u]]

    train_events_df = pd.DataFrame(train_events)
    if len(train_events_df) > 0:
        train_events_df = train_events_df.sort_values("timestamp", kind="mergesort")

    train_confidence = csr_int32_for_implicit(train_confidence)
    train_binary = csr_int32_for_implicit(train_binary)
    train_purchase_binary = csr_int32_for_implicit(train_purchase_binary)

    print("\nВременное разделение (последние ~20% событий пользователя → test):")
    print(f"  Train nnz: {train_binary.nnz}")
    print(f"  Пользователей с train и test: {len(test_users)}")
    print(f"  Train-событий с timestamp (для CV): {len(train_events_df)}")

    return (
        train_confidence,
        train_binary,
        train_purchase_binary,
        test_dict,
        test_users,
        train_events_df,
    )


def time_based_validation(
    train_events_df: pd.DataFrame,
    train_binary_full: csr_matrix,
    tune_user_indices: list[int],
    n_splits: int = N_CV_FOLDS,
    recommend_filter: str = "all_train",
    train_purchase_binary: csr_matrix | None = None,
    user_to_train_row_nnz: dict[int, int] | None = None,
):
    """
    Скользящее окно по времени на train-событиях: для каждого фолда глобально
    сортируем события по timestamp, делим на 60% / 20% / 20%, строим матрицу на 60%,
    метрики на пользователях из tune на ground truth из средних 20%.
    Сдвиг окна: offset = fold * (len // (4 * n_splits)) для разнообразия фолдов.
    """
    if train_events_df is None or len(train_events_df) < 10:
        raise ValueError("Недостаточно train-событий для time-based validation.")

    purchase_mat = train_purchase_binary
    n_users, n_items = train_binary_full.shape

    folds_metrics: list[dict[str, float]] = []
    fold_details: list[dict[str, Any]] = []

    ev = train_events_df.reset_index(drop=True)
    m = len(ev)
    step = max(1, m // (4 * max(n_splits, 1)))

    for fold in range(n_splits):
        offset = (fold * step) % max(1, m // 5)
        sub = ev.iloc[offset:].reset_index(drop=True)
        m2 = len(sub)
        if m2 < 20:
            sub = ev
            m2 = len(sub)
        cut_a = int(0.6 * m2)
        cut_b = int(0.8 * m2)
        cut_a = max(1, min(cut_a, m2 - 2))
        cut_b = max(cut_a + 1, min(cut_b, m2 - 1))

        part_train = sub.iloc[:cut_a]
        part_val = sub.iloc[cut_a:cut_b]

        val_dict: dict[int, set[int]] = defaultdict(set)
        for _, row in part_val.iterrows():
            val_dict[int(row["user_idx"])].add(int(row["item_idx"]))

        rows = part_train["user_idx"].astype(np.int32).tolist()
        cols = part_train["item_idx"].astype(np.int32).tolist()
        data = [1.0] * len(rows)
        train_fold = csr_matrix((data, (rows, cols)), shape=(n_users, n_items), dtype=np.float64)
        train_fold.sum_duplicates()
        train_fold = train_fold.tocsr()
        train_fold = csr_int32_for_implicit(train_fold)

        fold_details.append(
            {
                "fold": fold + 1,
                "offset": offset,
                "train_events": len(part_train),
                "val_events": len(part_val),
            }
        )
        folds_metrics.append(
            {
                "train_fold": train_fold,
                "val_dict": {u: list(s) for u, s in val_dict.items()},
            }
        )

    def filter_tune_users(train_fold: csr_matrix, val_dict: dict[int, list[int]], users: list[int]) -> list[int]:
        out = []
        for u in users:
            tr_n = int(train_fold[u].nnz)
            te_n = len(val_dict.get(u, []))
            if tr_n < 2 or te_n < 2:
                continue
            out.append(u)
        return out

    if user_to_train_row_nnz is None:
        user_to_train_row_nnz = {}
        for u in range(n_users):
            user_to_train_row_nnz[u] = int(train_binary_full[u].nnz)

    return folds_metrics, fold_details, user_to_train_row_nnz, filter_tune_users, purchase_mat


def evaluate_fast(
    recommender_func: Callable[[int], list[int]],
    user_indices: list[int],
    ground_truth: dict[int, list[int]],
    k_values: tuple[int, ...] = (5, 10),
    prediction_cache: dict[tuple[Any, ...], list[int]] | None = None,
    cache_key_extra: tuple[Any, ...] = (),
) -> dict[int, dict[str, float]]:
    results = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}

    for u in user_indices:
        actual = ground_truth.get(u)
        if not actual or len(actual) < 1:
            continue
        key = (u,) + cache_key_extra
        if prediction_cache is not None and key in prediction_cache:
            rec = prediction_cache[key]
        else:
            try:
                rec = recommender_func(u)
            except Exception as e:
                log.warning("evaluate_fast user %s: %s", u, e)
                continue
            if prediction_cache is not None:
                prediction_cache[key] = rec

        for k in k_values:
            results[k]["precision"].append(precision_at_k(rec, actual, k))
            results[k]["recall"].append(recall_at_k(rec, actual, k))
            results[k]["ndcg"].append(ndcg_at_k(rec, actual, k))

    out: dict[int, dict[str, float]] = {}
    for k in k_values:
        if not results[k]["precision"]:
            out[k] = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}
        else:
            out[k] = {
                "precision": float(np.mean(results[k]["precision"])),
                "recall": float(np.mean(results[k]["recall"])),
                "ndcg": float(np.mean(results[k]["ndcg"])),
            }
    return out


def _make_rec_fn(
    model,
    train_matrix: csr_matrix,
    popular_items: list[int],
    n_items: int,
    recommend_filter: str,
    train_purchase_binary: csr_matrix,
):
    def rec(uid: int):
        try:
            if recommend_filter == "purchases_only":
                uvec = train_purchase_binary[uid]
            else:
                uvec = train_matrix[uid]
            r = model.recommend(uid, uvec, N=n_items, filter_already_liked_items=True)
            return [int(x) for x in r[0]]
        except Exception:
            return popular_items

    return rec


def _eval_model_cv(
    model_builder: Callable[[], Any],
    folds_data: list[dict],
    tune_users: list[int],
    filter_tune_users,
    recommend_filter: str,
    train_purchase_binary: csr_matrix,
    popular_items: list[int],
    n_items: int = 10,
) -> tuple[float, dict[int, dict[str, float]], list[dict[int, dict[str, float]]]]:
    """Средний combined score по фолдам + список метрик по фолдам."""
    fold_mets: list[dict[int, dict[str, float]]] = []
    combined_scores: list[float] = []

    for fd in folds_data:
        train_fold = fd["train_fold"]
        val_dict = fd["val_dict"]
        model = model_builder()
        model.fit(train_fold, show_progress=False)

        users_fold = filter_tune_users(train_fold, val_dict, tune_users)
        if not users_fold:
            combined_scores.append(0.0)
            fold_mets.append({5: {"precision": 0, "recall": 0, "ndcg": 0}, 10: {"precision": 0, "recall": 0, "ndcg": 0}})
            continue

        rec_fn = _make_rec_fn(model, train_fold, popular_items, n_items, recommend_filter, train_purchase_binary)
        ev = evaluate_fast(rec_fn, users_fold, val_dict, k_values=(5, 10))
        p10, r10, n10 = ev[10]["precision"], ev[10]["recall"], ev[10]["ndcg"]
        combined_scores.append(combined_score(p10, r10, n10))
        fold_mets.append(ev)

    mean_combined = float(np.mean(combined_scores)) if combined_scores else 0.0
    # усреднённые метрики по фолдам (для отчёта)
    avg_ev: dict[int, dict[str, float]] = {}
    if fold_mets:
        for k in (5, 10):
            avg_ev[k] = {
                "precision": float(np.mean([fm[k]["precision"] for fm in fold_mets])),
                "recall": float(np.mean([fm[k]["recall"] for fm in fold_mets])),
                "ndcg": float(np.mean([fm[k]["ndcg"] for fm in fold_mets])),
            }
    else:
        avg_ev = {5: {"precision": 0, "recall": 0, "ndcg": 0}, 10: {"precision": 0, "recall": 0, "ndcg": 0}}

    return mean_combined, avg_ev, fold_mets


def _itemknn_eval_k_with_cache(
    K: int,
    approx: bool,
    nthr: int,
    sim_list: list[csr_matrix],
    folds_data: list[dict],
    tune_users: list[int],
    filter_tune_users,
    recommend_filter: str,
    train_purchase_binary: csr_matrix,
    popular_items: list[int],
) -> tuple[float, dict[int, dict[str, float]], list[dict[int, dict[str, float]]]]:
    """Оценка одного K на всех фолдах при уже посчитанных полных similarity (max K) на каждом фолде."""
    fold_mets: list[dict[int, dict[str, float]]] = []
    combined_scores: list[float] = []

    for fi, fd in enumerate(folds_data):
        train_fold = fd["train_fold"]
        val_dict = fd["val_dict"]
        sim_k = truncate_similarity_top_k(sim_list[fi], K)
        m = CosineRecommender(K=K, num_threads=nthr)
        if "approximate_similarities" in inspect.signature(CosineRecommender.__init__).parameters:
            m = _cosine_recommender_factory(K, nthr, approx)
        m.similarity = csr_int32_for_implicit(sim_k)
        m.K = K
        attach_scorer(m)

        users_fold = filter_tune_users(train_fold, val_dict, tune_users)
        if not users_fold:
            combined_scores.append(0.0)
            fold_mets.append({5: {"precision": 0, "recall": 0, "ndcg": 0}, 10: {"precision": 0, "recall": 0, "ndcg": 0}})
            continue

        rec_fn = _make_rec_fn(m, train_fold, popular_items, 10, recommend_filter, train_purchase_binary)
        ev = evaluate_fast(rec_fn, users_fold, val_dict, k_values=(5, 10))
        p10, r10, n10 = ev[10]["precision"], ev[10]["recall"], ev[10]["ndcg"]
        combined_scores.append(combined_score(p10, r10, n10))
        fold_mets.append(ev)

    mean_c = float(np.mean(combined_scores)) if combined_scores else 0.0
    avg_ev: dict[int, dict[str, float]] = {}
    if fold_mets:
        for k in (5, 10):
            avg_ev[k] = {
                "precision": float(np.mean([fm[k]["precision"] for fm in fold_mets])),
                "recall": float(np.mean([fm[k]["recall"] for fm in fold_mets])),
                "ndcg": float(np.mean([fm[k]["ndcg"] for fm in fold_mets])),
            }
    else:
        avg_ev = {5: {"precision": 0, "recall": 0, "ndcg": 0}, 10: {"precision": 0, "recall": 0, "ndcg": 0}}

    return mean_c, avg_ev, fold_mets


def grid_search_itemknn(
    folds_data: list[dict],
    tune_users: list[int],
    train_purchase_binary: csr_matrix,
    popular_items: list[int],
    filter_tune_users,
    recommend_filter: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    print("\n" + "=" * 60)
    print("ГЛУБОКИЙ ПОИСК ПАРАМЕТРОВ ДЛЯ ItemKNN (CosineRecommender)")
    print("=" * 60)

    max_k = max(K_GRID)
    results_rows: list[dict[str, Any]] = []

    for approx in ITEMKNN_APPROX_OPTS:
        for nthr in ITEMKNN_THREAD_OPTS:
            sim_per_fold: list[csr_matrix] = []
            print(f"\nКэш similarity по фолдам: max_K={max_k}, approx={approx}, num_threads={nthr}")
            for _, fd in enumerate(tqdm(folds_data, desc=f"Cosine fit folds (threads={nthr})", file=sys.stdout)):
                train_fold = fd["train_fold"]
                base = _cosine_recommender_factory(max_k, nthr, approx)
                base.fit(train_fold, show_progress=False)
                sim_per_fold.append(base.similarity)

            ordered_K = sorted(K_GRID)
            best_score_block = -float("inf")
            no_improve = 0

            for K in tqdm(ordered_K, desc=f"ItemKNN K sweep approx={approx} thr={nthr}", file=sys.stdout):
                mean_c, avg_ev, _ = _itemknn_eval_k_with_cache(
                    K,
                    approx,
                    nthr,
                    sim_per_fold,
                    folds_data,
                    tune_users,
                    filter_tune_users,
                    recommend_filter,
                    train_purchase_binary,
                    popular_items,
                )
                results_rows.append(
                    {
                        "model": "ItemKNN",
                        "K": K,
                        "approximate_similarities": approx,
                        "num_threads": nthr,
                        "k1": "",
                        "b": "",
                        "val_precision_10": avg_ev[10]["precision"],
                        "val_recall_10": avg_ev[10]["recall"],
                        "val_ndcg_10": avg_ev[10]["ndcg"],
                        "val_combined": mean_c,
                    }
                )
                print(
                    f"  K={K}, approx={approx}, threads={nthr} → "
                    f"P@10={avg_ev[10]['precision']:.4f}, R@10={avg_ev[10]['recall']:.4f}, "
                    f"NDCG@10={avg_ev[10]['ndcg']:.4f}, Combined={mean_c:.4f}"
                )

                if mean_c > best_score_block + EARLY_STOP_MIN_DELTA:
                    best_score_block = mean_c
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    log.info(
                        "ItemKNN early stop: approx=%s threads=%s после K без улучшения %s",
                        approx,
                        nthr,
                        EARLY_STOP_PATIENCE,
                    )
                    break

    df = pd.DataFrame(results_rows)
    if len(df) == 0:
        best = {}
    else:
        i = df["val_combined"].idxmax()
        best = df.loc[i].to_dict()

    print("\n" + "=" * 60)
    print("ЛУЧШИЕ ПАРАМЕТРЫ ItemKNN:")
    if best:
        print(f"  K={best.get('K')}, approx={best.get('approximate_similarities')}, threads={best.get('num_threads')}")
        print(
            f"  Val P@10={best.get('val_precision_10'):.4f}, R@10={best.get('val_recall_10'):.4f}, "
            f"NDCG@10={best.get('val_ndcg_10'):.4f}"
        )
    print("=" * 60)

    return best, df


def print_bm25_best_validation_detail(
    best: dict[str, Any],
    folds_data: list[dict],
    tune_users: list[int],
    filter_tune_users,
    recommend_filter: str,
    train_purchase_binary: csr_matrix,
    popular_items: list[int],
) -> None:
    if not best:
        return
    K = int(best["K"])
    k1 = float(best["k1"])
    b = float(best["b"])
    print("\n" + "=" * 60)
    print(f"Тестируется (лучшая по сетке): K={K}, k1={k1}, b={b}")
    print("=" * 60)
    precs, recalls, ndcgs, combined = [], [], [], []
    for fi, fd in enumerate(folds_data):
        model = BM25Recommender(K=K, K1=k1, B=b, num_threads=NUM_THREADS_DEFAULT)
        model.fit(fd["train_fold"], show_progress=False)
        users_fold = filter_tune_users(fd["train_fold"], fd["val_dict"], tune_users)
        rec_fn = _make_rec_fn(model, fd["train_fold"], popular_items, 10, recommend_filter, train_purchase_binary)
        ev = evaluate_fast(rec_fn, users_fold, fd["val_dict"], k_values=(5, 10))
        p10, r10, n10 = ev[10]["precision"], ev[10]["recall"], ev[10]["ndcg"]
        cmb = combined_score(p10, r10, n10)
        precs.append(p10)
        recalls.append(r10)
        ndcgs.append(n10)
        combined.append(cmb)
        print(
            f"  Валидация фолд {fi + 1}/{len(folds_data)}: P@10={p10:.4f}, R@10={r10:.4f}, NDCG@10={n10:.4f}"
        )
    print(
        f"  Среднее: P@10={float(np.mean(precs)):.4f}, R@10={float(np.mean(recalls)):.4f}, "
        f"NDCG@10={float(np.mean(ndcgs)):.4f}, Combined={float(np.mean(combined)):.4f}"
    )
    print("=" * 60)


def _bm25_eval_single(
    params: dict[str, Any],
    folds_data: list[dict],
    tune_users: list[int],
    filter_tune_users,
    recommend_filter: str,
    train_purchase_binary: csr_matrix,
    popular_items: list[int],
) -> dict[str, Any]:
    K = params["K"]
    k1 = params["k1"]
    b = params["b"]

    def builder():
        return BM25Recommender(K=K, K1=k1, B=b, num_threads=NUM_THREADS_DEFAULT)

    mean_c, avg_ev, _ = _eval_model_cv(
        builder,
        folds_data,
        tune_users,
        filter_tune_users,
        recommend_filter,
        train_purchase_binary,
        popular_items,
    )
    return {
        "model": "BM25",
        "K": K,
        "k1": k1,
        "b": b,
        "approximate_similarities": "",
        "num_threads": NUM_THREADS_DEFAULT,
        "val_precision_10": avg_ev[10]["precision"],
        "val_recall_10": avg_ev[10]["recall"],
        "val_ndcg_10": avg_ev[10]["ndcg"],
        "val_combined": mean_c,
    }


def grid_search_bm25(
    folds_data: list[dict],
    tune_users: list[int],
    filter_tune_users,
    recommend_filter: str,
    train_purchase_binary: csr_matrix,
    popular_items: list[int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    print("\n" + "=" * 60)
    print("ГЛУБОКИЙ ПОИСК ПАРАМЕТРОВ ДЛЯ BM25")
    print("=" * 60)

    grid = [{"K": k, "k1": k1, "b": b} for k, k1, b in product(K_GRID, BM25_K1_GRID, BM25_B_GRID)]

    def job(p):
        return _bm25_eval_single(
            p,
            folds_data,
            tune_users,
            filter_tune_users,
            recommend_filter,
            train_purchase_binary,
            popular_items,
        )

    rows = Parallel(n_jobs=-1)(delayed(job)(p) for p in tqdm(grid, desc="BM25 grid (joblib)", file=sys.stdout))
    df = pd.DataFrame(rows)

    for _, r in df.sort_values("val_combined", ascending=False).head(5).iterrows():
        print(
            f"  K={r['K']}, k1={r['k1']}, b={r['b']} → Combined={r['val_combined']:.4f}, "
            f"P@10={r['val_precision_10']:.4f}"
        )

    best = df.loc[df["val_combined"].idxmax()].to_dict() if len(df) else {}
    print_bm25_best_validation_detail(
        best,
        folds_data,
        tune_users,
        filter_tune_users,
        recommend_filter,
        train_purchase_binary,
        popular_items,
    )
    print("\n" + "=" * 60)
    print("ЛУЧШИЕ ПАРАМЕТРЫ BM25:")
    if best:
        print(f"  K={best.get('K')}, k1={best.get('k1')}, b={best.get('b')}")
        print(
            f"  Val P@10={best.get('val_precision_10'):.4f}, R@10={best.get('val_recall_10'):.4f}, "
            f"NDCG@10={best.get('val_ndcg_10'):.4f}"
        )
    print("=" * 60)

    return best, df


def _tfidf_eval_single(
    params: dict[str, Any],
    folds_data: list[dict],
    tune_users: list[int],
    filter_tune_users,
    recommend_filter: str,
    train_purchase_binary: csr_matrix,
    popular_items: list[int],
) -> dict[str, Any]:
    K = params["K"]

    def builder():
        return TFIDFRecommender(K=K, num_threads=NUM_THREADS_DEFAULT)

    mean_c, avg_ev, _ = _eval_model_cv(
        builder,
        folds_data,
        tune_users,
        filter_tune_users,
        recommend_filter,
        train_purchase_binary,
        popular_items,
    )
    return {
        "model": "TFIDF",
        "K": K,
        "k1": "",
        "b": "",
        "approximate_similarities": "",
        "num_threads": NUM_THREADS_DEFAULT,
        "val_precision_10": avg_ev[10]["precision"],
        "val_recall_10": avg_ev[10]["recall"],
        "val_ndcg_10": avg_ev[10]["ndcg"],
        "val_combined": mean_c,
    }


def grid_search_tfidf(
    folds_data: list[dict],
    tune_users: list[int],
    filter_tune_users,
    recommend_filter: str,
    train_purchase_binary: csr_matrix,
    popular_items: list[int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    print("\n" + "=" * 60)
    print("ПОИСК ПАРАМЕТРОВ TF-IDF (TFIDFRecommender)")
    print("=" * 60)

    grid = [{"K": k} for k in K_GRID]

    def job(p):
        return _tfidf_eval_single(
            p,
            folds_data,
            tune_users,
            filter_tune_users,
            recommend_filter,
            train_purchase_binary,
            popular_items,
        )

    rows = Parallel(n_jobs=-1)(delayed(job)(p) for p in tqdm(grid, desc="TFIDF grid", file=sys.stdout))
    df = pd.DataFrame(rows)
    best = df.loc[df["val_combined"].idxmax()].to_dict() if len(df) else {}
    print("\nЛУЧШИЕ TF-IDF:", best.get("K"), "Combined=", best.get("val_combined"))
    return best, df


def plot_results(
    df_item: pd.DataFrame,
    df_bm25: pd.DataFrame,
    df_tfidf: pd.DataFrame,
    comparison_df: pd.DataFrame,
):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    sns.set(style="whitegrid", font_scale=1.0)

    # Метрики от K — line plot (по валидации)
    plt.figure(figsize=(10, 6))
    for name, dfi, key in [
        ("ItemKNN", df_item, "K"),
        ("BM25", df_bm25, "K"),
        ("TFIDF", df_tfidf, "K"),
    ]:
        if dfi is None or len(dfi) == 0:
            continue
        g = dfi.groupby(key)["val_combined"].max().reset_index()
        plt.plot(g[key], g["val_combined"], marker="o", label=name)
    plt.xlabel("K")
    plt.ylabel("Val combined score")
    plt.title("Combined validation score vs K (max over other params)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "metrics_vs_K.png"), dpi=150)
    plt.close()

    # Bar chart сравнение моделей (тест)
    if comparison_df is not None and len(comparison_df) > 0:
        plt.figure(figsize=(8, 5))
        labs = comparison_df["Модель"].tolist()
        x = np.arange(len(labs))
        plt.bar(x, comparison_df["combined_test_10"].astype(float))
        plt.xticks(x, labs, rotation=15)
        plt.ylabel("Combined@test (R,N,P @10)")
        plt.title("Модели: комбинированный скор на тесте")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "model_comparison_bar.png"), dpi=150)
        plt.close()

    # Тепловая карта BM25: k1 vs b (усреднение по K)
    if df_bm25 is not None and len(df_bm25) > 0:
        pivot = df_bm25.groupby(["k1", "b"])["val_combined"].mean().unstack()
        plt.figure(figsize=(7, 5))
        sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
        plt.title("BM25: средний val_combined по K (k1 vs b)")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "bm25_k1_b_heatmap.png"), dpi=150)
        plt.close()

    log.info("Графики сохранены в %s", PLOTS_DIR)


def train_final_model(model_name: str, best: dict[str, Any], train_binary: csr_matrix) -> Any:
    if model_name == "ItemKNN":
        K = int(best["K"])
        nt = int(best.get("num_threads") or NUM_THREADS_DEFAULT)
        raw_apx = best.get("approximate_similarities", False)
        if isinstance(raw_apx, str) and raw_apx == "":
            use_apx = False
        else:
            use_apx = bool(raw_apx)
        m = _cosine_recommender_factory(K, nt, use_apx)
        m.fit(train_binary, show_progress=True)
        return m
    if model_name == "BM25":
        m = BM25Recommender(
            K=int(best["K"]),
            K1=float(best["k1"]),
            B=float(best["b"]),
            num_threads=NUM_THREADS_DEFAULT,
        )
        m.fit(train_binary, show_progress=True)
        return m
    if model_name == "TFIDF":
        m = TFIDFRecommender(K=int(best["K"]), num_threads=NUM_THREADS_DEFAULT)
        m.fit(train_binary, show_progress=True)
        return m
    raise ValueError(model_name)


def main(recommend_filter: str = "all_train"):
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    if recommend_filter not in ("all_train", "purchases_only"):
        raise ValueError("recommend_filter должен быть 'all_train' или 'purchases_only'")

    print("=" * 60)
    print("BM25 / ItemKNN / TF-IDF — подбор гиперпараметров (RetailRocket)")
    print("=" * 60)

    data = load_data()
    train_confidence, train_binary, train_purchase_binary, test_dict, test_users, train_events_df = train_test_split_temporal(
        data, test_ratio=0.2
    )
    _ = train_confidence  # матрица уверенности не нужна для KNN/BM25

    popular_items = get_popular_items(train_binary, n_items=10)
    rng = np.random.default_rng(42)
    tune_pool = list(test_users)
    if len(tune_pool) > 400:
        tune_pool = [tune_pool[i] for i in rng.choice(len(tune_pool), size=400, replace=False)]
    tune_users = sorted(tune_pool)

    user_to_train_row_nnz = {u: int(train_binary[u].nnz) for u in range(train_binary.shape[0])}

    folds_raw, fold_details, _, filter_tune_users, _ = time_based_validation(
        train_events_df,
        train_binary,
        tune_users,
        n_splits=N_CV_FOLDS,
        recommend_filter=recommend_filter,
        train_purchase_binary=train_purchase_binary,
        user_to_train_row_nnz=user_to_train_row_nnz,
    )

    print("\nФолды time-based validation:", json.dumps(fold_details, ensure_ascii=False))

    best_item, df_item = grid_search_itemknn(
        folds_raw,
        tune_users,
        train_purchase_binary,
        popular_items,
        filter_tune_users,
        recommend_filter,
    )

    best_bm25, df_bm25 = grid_search_bm25(
        folds_raw,
        tune_users,
        filter_tune_users,
        recommend_filter,
        train_purchase_binary,
        popular_items,
    )

    best_tfidf, df_tfidf = grid_search_tfidf(
        folds_raw,
        tune_users,
        filter_tune_users,
        recommend_filter,
        train_purchase_binary,
        popular_items,
    )

    all_rows = []
    for d in (df_item, df_bm25, df_tfidf):
        if d is not None and len(d) > 0:
            all_rows.append(d)
    grid_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    grid_all.to_csv(os.path.join(DATA_DIR, "grid_search_results.csv"), index=False)
    log.info("Сохранено %s", os.path.join(DATA_DIR, "grid_search_results.csv"))

    # Финальное обучение лучших и тест
    def eval_test(model, name: str) -> dict[str, Any]:
        rec_fn = _make_rec_fn(model, train_binary, popular_items, 10, recommend_filter, train_purchase_binary)
        ev = evaluate_fast(rec_fn, test_users, test_dict, k_values=(5, 10))
        c10 = combined_score(ev[10]["precision"], ev[10]["recall"], ev[10]["ndcg"])
        return {
            "Модель": name,
            "K": getattr(model, "K", ""),
            "P@5": f"{ev[5]['precision']:.4f}",
            "R@5": f"{ev[5]['recall']:.4f}",
            "NDCG@5": f"{ev[5]['ndcg']:.4f}",
            "P@10": f"{ev[10]['precision']:.4f}",
            "R@10": f"{ev[10]['recall']:.4f}",
            "NDCG@10": f"{ev[10]['ndcg']:.4f}",
            "combined_test_10": c10,
        }

    models_to_save = []
    if best_item:
        m_i = train_final_model("ItemKNN", best_item, train_binary)
        models_to_save.append(("ItemKNN", m_i, best_item))

    if best_bm25:
        m_b = BM25Recommender(
            K=int(best_bm25["K"]),
            K1=float(best_bm25["k1"]),
            B=float(best_bm25["b"]),
            num_threads=NUM_THREADS_DEFAULT,
        )
        m_b.fit(train_binary, show_progress=True)
        models_to_save.append(("BM25", m_b, best_bm25))

    if best_tfidf:
        m_t = TFIDFRecommender(K=int(best_tfidf["K"]), num_threads=NUM_THREADS_DEFAULT)
        m_t.fit(train_binary, show_progress=True)
        models_to_save.append(("TFIDF", m_t, best_tfidf))

    comparison_rows = []
    for label, model, _ in models_to_save:
        comparison_rows.append(eval_test(model, label))

    comparison_df = pd.DataFrame(comparison_rows)
    if len(comparison_df) > 0:
        comparison_df.to_csv(os.path.join(DATA_DIR, "model_comparison_bm25_itemknn.csv"), index=False)

    print("\n" + "=" * 60)
    print("ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ (на тесте)")
    print("=" * 60)
    if len(comparison_df) > 0:
        disp = comparison_df.drop(columns=["combined_test_10"], errors="ignore")
        print(disp.to_string(index=False))
        best_row = comparison_df.loc[comparison_df["combined_test_10"].astype(float).idxmax()]
        print(f"\nЛУЧШАЯ МОДЕЛЬ: {best_row['Модель']} (Combined@test: {float(best_row['combined_test_10']):.4f})")

    plot_results(df_item, df_bm25, df_tfidf, comparison_df)

    # Сохранение моделей
    path_map = {
        "ItemKNN": os.path.join(DATA_DIR, "best_itemknn.pkl"),
        "BM25": os.path.join(DATA_DIR, "best_bm25.pkl"),
        "TFIDF": os.path.join(DATA_DIR, "best_tfidf.pkl"),
    }
    for label, model, _ in models_to_save:
        with open(path_map[label], "wb") as f:
            pickle.dump(model, f)
        log.info("Сохранена модель %s → %s", label, path_map[label])

    report_path = os.path.join(DATA_DIR, "hyperparameter_tuning_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Лучшие параметры (валидация time-based, combined = 0.5*R@10+0.3*NDCG@10+0.2*P@10)\n")
        f.write(json.dumps({"ItemKNN": best_item, "BM25": best_bm25, "TFIDF": best_tfidf}, ensure_ascii=False, indent=2))
        f.write("\n\nФолды CV:\n")
        f.write(json.dumps(fold_details, ensure_ascii=False, indent=2))
        f.write("\n")
    log.info("Отчёт: %s", report_path)

    return {
        "best_item": best_item,
        "best_bm25": best_bm25,
        "best_tfidf": best_tfidf,
        "comparison": comparison_df,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--recommend-filter", choices=["all_train", "purchases_only"], default="all_train")
    args = p.parse_args()
    main(recommend_filter=args.recommend_filter)
