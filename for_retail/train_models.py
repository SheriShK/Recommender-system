"""
Обучение ALS / BPR / ItemKNN на RetailRocket.
Сплит: последние 20% событий каждого пользователя (по timestamp) → test.
Рекомендации: только user_id, item_id, relevance; без признаков и эмбеддингов.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import CosineRecommender

DATA_DIR = "data"

# implicit 0.7.2 на Windows: баг int/long в all_pairs_knn (внутренние rows/cols).
# Решение: pip install --force-reinstall "git+https://github.com/benfred/implicit.git"
ITEMKNN_WINDOWS_FIX_HINT = (
    "ItemKNN: на Windows с implicit 0.7.2 из PyPI известен сбой (long vs long long). "
    "Поставьте сборку с GitHub: "
    "pip install --force-reinstall \"git+https://github.com/benfred/implicit.git\" "
    "либо запустите: python train_models.py --skip-itemknn"
)


def csr_int32_for_implicit(mat: csr_matrix) -> csr_matrix:
    """
    На Windows implicit (Cython KNN) ожидает indptr/indices как int32.
    Иначе: ValueError: Buffer dtype mismatch, expected 'long' but got 'long long'.
    """
    mat = mat.tocsr(copy=True)
    mat.indptr = np.asarray(mat.indptr, dtype=np.int32)
    mat.indices = np.asarray(mat.indices, dtype=np.int32)
    if mat.data.size > 0:
        mat.data = np.asarray(mat.data, dtype=np.float64)
    return mat


# ========== 1. ЗАГРУЗКА ДАННЫХ ==========

def load_data():
    """Загружает items, interactions, матрицы и маппинги из папки data/."""
    print("Загрузка данных...")

    items_df = pd.read_csv(f"{DATA_DIR}/items.csv")
    interactions_df = pd.read_csv(f"{DATA_DIR}/interactions.csv")

    interaction_matrix = load_npz(f"{DATA_DIR}/interaction_matrix.npz")
    confidence_matrix = load_npz(f"{DATA_DIR}/confidence_matrix.npz")

    with open(f"{DATA_DIR}/user_to_idx.json", "r", encoding="utf-8") as f:
        user_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
    with open(f"{DATA_DIR}/item_to_idx.json", "r", encoding="utf-8") as f:
        item_to_idx = {int(k): int(v) for k, v in json.load(f).items()}

    idx_to_user = {v: int(k) for k, v in user_to_idx.items()}
    idx_to_item = {v: int(k) for k, v in item_to_idx.items()}

    print(
        f"[OK] Загружено: {interaction_matrix.shape[0]} пользователей, "
        f"{interaction_matrix.shape[1]} товаров"
    )
    print(f"[OK] Ненулевых взаимодействий (агрегированных): {interaction_matrix.nnz}")

    return {
        "items": items_df,
        "interactions": interactions_df,
        "interaction_matrix": interaction_matrix,
        "confidence_matrix": confidence_matrix,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_user": idx_to_user,
        "idx_to_item": idx_to_item,
    }


# ========== 2. МЕТРИКИ ==========

def precision_at_k(recommended, actual, k):
    recommended_k = recommended[:k]
    if len(actual) == 0:
        return 0.0
    hits = len(set(recommended_k) & set(actual))
    return hits / k


def recall_at_k(recommended, actual, k):
    recommended_k = recommended[:k]
    if len(actual) == 0:
        return 0.0
    hits = len(set(recommended_k) & set(actual))
    return hits / len(actual)


def ndcg_at_k(recommended, actual, k):
    recommended_k = recommended[:k]
    if len(actual) == 0:
        return 0.0

    relevance = [1 if item in actual else 0 for item in recommended_k]

    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance) if rel > 0)

    ideal = [1] * min(len(actual), k) + [0] * (k - min(len(actual), k))
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal) if rel > 0)

    return dcg / idcg if idcg > 0 else 0.0


def get_popular_items(train_matrix, n_items=10):
    item_popularity = np.array(train_matrix.sum(axis=0)).flatten()
    return np.argsort(item_popularity)[-n_items:][::-1].tolist()


def evaluate_model(model_name, recommender_func, test_users, test_dict, k_values=(5, 10)):
    results = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}

    for user_idx in test_users:
        try:
            recommended_items_idx = recommender_func(user_idx)
        except Exception as e:
            print(f"  Ошибка для пользователя {user_idx}: {e}")
            continue

        actual_items_idx = test_dict.get(user_idx, [])
        if len(actual_items_idx) == 0:
            continue

        for k in k_values:
            results[k]["precision"].append(precision_at_k(recommended_items_idx, actual_items_idx, k))
            results[k]["recall"].append(recall_at_k(recommended_items_idx, actual_items_idx, k))
            results[k]["ndcg"].append(ndcg_at_k(recommended_items_idx, actual_items_idx, k))

    avg_results = {}
    for k in k_values:
        if len(results[k]["precision"]) == 0:
            avg_results[k] = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}
        else:
            avg_results[k] = {
                "precision": float(np.mean(results[k]["precision"])),
                "recall": float(np.mean(results[k]["recall"])),
                "ndcg": float(np.mean(results[k]["ndcg"])),
            }
    return avg_results


# ========== 3. ОБУЧЕНИЕ МОДЕЛЕЙ ==========

def train_als(train_confidence, factors=50, iterations=20, regularization=0.1):
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ALS (Alternating Least Squares)")
    print("=" * 60)

    model = AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=42,
        num_threads=4,
    )
    print(f"Параметры: factors={factors}, iterations={iterations}, regularization={regularization}")
    print(f"Размер матрицы: {train_confidence.shape}")
    nnz = train_confidence.nnz
    n = train_confidence.shape[0] * train_confidence.shape[1]
    print(f"Плотность: {nnz / max(1, n) * 100:.4f}%")

    model.fit(train_confidence)
    print("[OK] Модель обучена")
    return model


def train_bpr(train_binary, factors=50, iterations=20, learning_rate=0.01):
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ BPR (Bayesian Personalised Ranking)")
    print("=" * 60)

    model = BayesianPersonalizedRanking(
        factors=factors,
        iterations=iterations,
        learning_rate=learning_rate,
        random_state=42,
        num_threads=4,
    )
    print(f"Параметры: factors={factors}, iterations={iterations}, learning_rate={learning_rate}")
    print(f"Размер матрицы: {train_binary.shape}")

    model.fit(train_binary)
    print("[OK] Модель обучена")
    return model


def train_itemknn(train_binary, k_neighbors=20):
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ ItemKNN (Item-based Collaborative Filtering)")
    print("=" * 60)

    model = CosineRecommender(K=k_neighbors, num_threads=4)
    print(f"Параметры: K_neighbors={k_neighbors} (cosine similarity)")
    print(f"Размер матрицы: {train_binary.shape}")
    nnz = train_binary.nnz
    n = train_binary.shape[0] * train_binary.shape[1]
    print(f"Плотность: {nnz / max(1, n) * 100:.4f}%")

    model.fit(train_binary)
    print("[OK] Модель ItemKNN обучена")
    return model


# ========== 4. ПОДБОР ГИПЕРПАРАМЕТРОВ ==========

def _safe_precision(metrics):
    p = metrics[10]["precision"]
    return 0.0 if (np.isnan(p) or np.isinf(p)) else p


def _make_recommender_als(model, train_confidence, popular_items, n_items, filter_mask, filter_purchases_mask):
    """
    filter_mode передаётся через closure: user_items для implicit.recommend.
    - all_train: маска всех train-взаимодействий
    - purchases_only: только покупки в train (фильтровать уже купленное)
    """

    def rec(uid, _m=model):
        try:
            if filter_mask == "purchases_only":
                uvec = filter_purchases_mask[uid]
            else:
                uvec = train_confidence[uid]
            r = _m.recommend(uid, uvec, N=n_items, filter_already_liked_items=True)
            return [int(x) for x in r[0]]
        except Exception:
            return popular_items

    return rec


def _make_recommender_bpr_knn(model, train_binary, popular_items, n_items, filter_mask, filter_purchases_mask):
    def rec(uid, _m=model):
        try:
            if filter_mask == "purchases_only":
                uvec = filter_purchases_mask[uid]
            else:
                uvec = train_binary[uid]
            r = _m.recommend(uid, uvec, N=n_items, filter_already_liked_items=True)
            return [int(x) for x in r[0]]
        except Exception:
            return popular_items

    return rec


def tune_als(
    train_confidence,
    train_purchase_binary,
    popular_items,
    tune_users,
    test_dict,
    factors_list,
    iterations_list,
    regularization_list,
    n_items=10,
    recommend_filter="all_train",
):
    best_precision, best_params, best_model = -1.0, None, None

    for factors in factors_list:
        for iterations in iterations_list:
            for reg in regularization_list:
                print(f"  ALS grid: factors={factors}, iter={iterations}, reg={reg}")
                model = train_als(train_confidence, factors, iterations, reg)

                rec = _make_recommender_als(
                    model,
                    train_confidence,
                    popular_items,
                    n_items,
                    recommend_filter,
                    train_purchase_binary,
                )
                p10 = _safe_precision(evaluate_model("ALS", rec, tune_users, test_dict, k_values=[10]))
                print(f"    Precision@10 (tune): {p10:.4f}")

                if best_model is None or p10 > best_precision:
                    best_precision = p10
                    best_params = {"factors": factors, "iterations": iterations, "regularization": reg}
                    best_model = model

    return best_model, best_params, best_precision


def tune_bpr(
    train_binary,
    train_purchase_binary,
    popular_items,
    tune_users,
    test_dict,
    factors_list,
    iterations_list,
    lr_list,
    n_items=10,
    recommend_filter="all_train",
):
    best_precision, best_params, best_model = -1.0, None, None

    for factors in factors_list:
        for iterations in iterations_list:
            for lr in lr_list:
                print(f"  BPR grid: factors={factors}, iter={iterations}, lr={lr}")
                model = train_bpr(train_binary, factors, iterations, lr)

                rec = _make_recommender_bpr_knn(
                    model,
                    train_binary,
                    popular_items,
                    n_items,
                    recommend_filter,
                    train_purchase_binary,
                )
                p10 = _safe_precision(evaluate_model("BPR", rec, tune_users, test_dict, k_values=[10]))
                print(f"    Precision@10 (tune): {p10:.4f}")

                if best_model is None or p10 > best_precision:
                    best_precision = p10
                    best_params = {"factors": factors, "iterations": iterations, "learning_rate": lr}
                    best_model = model

    return best_model, best_params, best_precision


def tune_itemknn(
    train_binary,
    train_purchase_binary,
    popular_items,
    tune_users,
    test_dict,
    k_neighbors_list,
    n_items=10,
    recommend_filter="all_train",
):
    best_precision, best_params, best_model = -1.0, None, None

    for k in k_neighbors_list:
        print(f"  ItemKNN grid: k_neighbors={k}")
        model = train_itemknn(train_binary, k)

        rec = _make_recommender_bpr_knn(
            model,
            train_binary,
            popular_items,
            n_items,
            recommend_filter,
            train_purchase_binary,
        )
        p10 = _safe_precision(evaluate_model("ItemKNN", rec, tune_users, test_dict, k_values=[10]))
        print(f"    Precision@10 (tune): {p10:.4f}")

        if best_model is None or p10 > best_precision:
            best_precision = p10
            best_params = {"k_neighbors": k}
            best_model = model

    return best_model, best_params, best_precision


# ========== 5. ВРЕМЕННОЕ РАЗДЕЛЕНИЕ (по пользователю) ==========

def train_test_split(data, test_ratio=0.2):
    """
    Для каждого пользователя: сортировка по timestamp (Unix ms),
    последние ceil(n * ratio) событий (но не все) → test.
    Train-матрица: сумма relevance по парам (user, item) на train-отрезке.
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

    test_dict = {}

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
            continue

        n_test = max(1, min(int(np.ceil(n * test_ratio)), n - 1))
        test_idx_set = set(idxs[-n_test:])
        test_items_set = set()

        for idx, row in g.iterrows():
            i_idx = item_to_idx[int(row["item_id"])]
            rel = float(row["relevance"])
            if idx in test_idx_set:
                test_items_set.add(i_idx)
            else:
                tr_rows.append(u_idx)
                tr_cols.append(i_idx)
                tr_conf.append(rel)
                tr_bin.append(1.0)
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

    test_users = [
        u for u in test_dict if train_confidence.indptr[u + 1] > train_confidence.indptr[u]
    ]

    n_train = train_confidence.nnz
    n_test = sum(len(v) for v in test_dict.values())
    print("\nВременное разделение (последние ~20% событий пользователя → test):")
    print(f"  Train (ненулей в sparse): {n_train}")
    print(f"  Test (уникальных item на пользователя в сумме): {n_test}")
    print(f"  Пользователей с непустым train и test: {len(test_users)}")

    train_confidence = csr_int32_for_implicit(train_confidence)
    train_binary = csr_int32_for_implicit(train_binary)
    train_purchase_binary = csr_int32_for_implicit(train_purchase_binary)

    save_npz(f"{DATA_DIR}/train_interaction_matrix.npz", train_binary)
    save_npz(f"{DATA_DIR}/train_confidence_matrix.npz", train_confidence)
    save_npz(f"{DATA_DIR}/train_purchase_matrix.npz", train_purchase_binary)

    return train_confidence, train_binary, train_purchase_binary, test_dict, test_users


# ========== 6. СЭМПЛ ПОЛЬЗОВАТЕЛЕЙ ДЛЯ ГРИДА ==========

def sample_users(users, max_users=120, seed=42):
    rng = np.random.default_rng(seed)
    users = list(users)
    if len(users) <= max_users:
        return users
    idx = rng.choice(len(users), size=max_users, replace=False)
    return [users[i] for i in idx]


# ========== 7. MAIN ==========

def main(
    recommend_filter: str = "all_train",
    skip_itemknn: bool = False,
):
    """
    recommend_filter:
      - 'all_train' — не рекомендовать товары из train-истории пользователя
      - 'purchases_only' — скрывать только уже купленные в train (просмотры можно снова)
    """
    print("=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ (RetailRocket)")
    print("=" * 60)

    if recommend_filter not in ("all_train", "purchases_only"):
        raise ValueError("recommend_filter должен быть 'all_train' или 'purchases_only'")

    data = load_data()

    train_confidence, train_binary, train_purchase_binary, test_dict, test_users = train_test_split(
        data, test_ratio=0.2
    )

    popular_items = get_popular_items(train_binary, n_items=10)
    tune_users = sample_users(test_users, max_users=120, seed=42)
    print(f"\nПодбор гиперпараметров по {len(tune_users)} пользователям из теста...")
    print(f"Режим фильтрации при рекомендации: {recommend_filter}")

    results = {}
    best_params = {}

    factors_list = [20, 50]
    iterations_list = [10, 20]
    regularization_list = [0.01, 0.1]
    lr_list = [0.005, 0.01]
    k_neighbors_list = [10, 20, 50]

    print("\n" + "=" * 60)
    print("1. ПОДБОР ГИПЕРПАРАМЕТРОВ ALS")
    print("=" * 60)

    als_model, best_als, _ = tune_als(
        train_confidence,
        train_purchase_binary,
        popular_items,
        tune_users,
        test_dict,
        factors_list,
        iterations_list,
        regularization_list,
        recommend_filter=recommend_filter,
    )
    print(f"Лучшие параметры ALS: {best_als}")
    best_params["ALS"] = best_als

    als_rec = _make_recommender_als(
        als_model,
        train_confidence,
        popular_items,
        10,
        recommend_filter,
        train_purchase_binary,
    )
    results["ALS"] = evaluate_model("ALS", als_rec, test_users, test_dict, k_values=[5, 10])

    with open(f"{DATA_DIR}/model_als.pkl", "wb") as f:
        pickle.dump(als_model, f)
    print("[OK] ALS модель сохранена")

    print("\n" + "=" * 60)
    print("2. ПОДБОР ГИПЕРПАРАМЕТРОВ BPR")
    print("=" * 60)

    bpr_model, best_bpr, _ = tune_bpr(
        train_binary,
        train_purchase_binary,
        popular_items,
        tune_users,
        test_dict,
        factors_list,
        iterations_list,
        lr_list,
        recommend_filter=recommend_filter,
    )
    print(f"Лучшие параметры BPR: {best_bpr}")
    best_params["BPR"] = best_bpr

    bpr_rec = _make_recommender_bpr_knn(
        bpr_model,
        train_binary,
        popular_items,
        10,
        recommend_filter,
        train_purchase_binary,
    )
    results["BPR"] = evaluate_model("BPR", bpr_rec, test_users, test_dict, k_values=[5, 10])

    with open(f"{DATA_DIR}/model_bpr.pkl", "wb") as f:
        pickle.dump(bpr_model, f)
    print("[OK] BPR модель сохранена")

    print("\n" + "=" * 60)
    print("3. ПОДБОР ГИПЕРПАРАМЕТРОВ ItemKNN")
    print("=" * 60)

    if skip_itemknn:
        print("Пропуск ItemKNN (--skip-itemknn).")
    else:
        try:
            knn_model, best_knn, _ = tune_itemknn(
                train_binary,
                train_purchase_binary,
                popular_items,
                tune_users,
                test_dict,
                k_neighbors_list,
                recommend_filter=recommend_filter,
            )
            print(f"Лучшие параметры ItemKNN: {best_knn}")
            best_params["ItemKNN"] = best_knn

            knn_rec = _make_recommender_bpr_knn(
                knn_model,
                train_binary,
                popular_items,
                10,
                recommend_filter,
                train_purchase_binary,
            )
            results["ItemKNN"] = evaluate_model("ItemKNN", knn_rec, test_users, test_dict, k_values=[5, 10])

            with open(f"{DATA_DIR}/model_itemknn.pkl", "wb") as f:
                pickle.dump(knn_model, f)
            print("[OK] ItemKNN модель сохранена")
        except ValueError as e:
            if "dtype mismatch" in str(e) or "Buffer dtype" in str(e):
                print("\n[ОШИБКА ItemKNN]", e)
                print(ITEMKNN_WINDOWS_FIX_HINT)
            else:
                raise

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ АЛГОРИТМОВ")
    print("=" * 60)

    results_df = []
    for model_name, metrics in results.items():
        for k, km in metrics.items():
            results_df.append(
                {
                    "Модель": model_name,
                    "K": k,
                    "Precision@K": f"{km['precision']:.4f}",
                    "Recall@K": f"{km['recall']:.4f}",
                    "NDCG@K": f"{km['ndcg']:.4f}",
                }
            )

    results_table = pd.DataFrame(results_df)
    print(results_table.to_string(index=False))

    results_table.to_csv(f"{DATA_DIR}/model_comparison_results.csv", index=False)
    print("\n[OK] Результаты сохранены в data/model_comparison_results.csv")

    with open(f"{DATA_DIR}/best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump({**best_params, "recommend_filter": recommend_filter}, f, ensure_ascii=False, indent=2)
    print("[OK] Гиперпараметры сохранены в data/best_hyperparameters.json")

    best_precision = -1.0
    best_model = None
    for model_name, metrics in results.items():
        if 10 in metrics and metrics[10]["precision"] > best_precision:
            best_precision = metrics[10]["precision"]
            best_model = model_name

    print("\n" + "=" * 60)
    print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model}")
    print(f"   Precision@10: {best_precision:.4f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recommend-filter",
        choices=["all_train", "purchases_only"],
        default="all_train",
        help="Что исключать при рекомендации: все train-события или только покупки",
    )
    parser.add_argument(
        "--skip-itemknn",
        action="store_true",
        help="Не обучать ItemKNN (ускорение / обход сбоя implicit на Windows)",
    )
    args = parser.parse_args()
    main(recommend_filter=args.recommend_filter, skip_itemknn=args.skip_itemknn)
