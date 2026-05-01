# Оценка моделей, сравнение с Popularity, графики, рекомендации (RetailRocket: category_id).

from __future__ import annotations

import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import load_npz

from train_models import (
    load_data,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    train_test_split,
)

sns.set(style="whitegrid", font_scale=1.2)

DATA_DIR = "data"

DEFAULT_ITEM_DESCRIPTION = "Товар RetailRocket (нет текстового описания в датасете)"


def _load_recommend_filter_config() -> str:
    path = f"{DATA_DIR}/best_hyperparameters.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("recommend_filter", "all_train")
    except OSError:
        return "all_train"


def load_models():
    """Загружает сохранённые модели и маппинги."""
    print("Загрузка данных и моделей...")

    with open(f"{DATA_DIR}/user_to_idx.json", "r", encoding="utf-8") as f:
        user_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
    with open(f"{DATA_DIR}/item_to_idx.json", "r", encoding="utf-8") as f:
        item_to_idx = {int(k): int(v) for k, v in json.load(f).items()}

    idx_to_user = {v: k for k, v in user_to_idx.items()}
    idx_to_item = {v: k for k, v in item_to_idx.items()}

    items_df = pd.read_csv(f"{DATA_DIR}/items.csv")

    models = {}
    for name, path in {
        "ALS": f"{DATA_DIR}/model_als.pkl",
        "BPR": f"{DATA_DIR}/model_bpr.pkl",
        "ItemKNN": f"{DATA_DIR}/model_itemknn.pkl",
    }.items():
        try:
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        except OSError:
            print(f"  Модель {name} не найдена")

    print(f"[OK] Загружено {len(models)} моделей, {len(user_to_idx)} пользователей")

    return {
        "items": items_df,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_user": idx_to_user,
        "idx_to_item": idx_to_item,
    }, models


def get_train_user_row(model_name: str, user_idx: int):
    if model_name == "ALS":
        return load_npz(f"{DATA_DIR}/train_confidence_matrix.npz")[user_idx]
    return load_npz(f"{DATA_DIR}/train_interaction_matrix.npz")[user_idx]


def recommend_for_user(
    user_id: int,
    model_name: str = "ALS",
    n_items: int = 10,
    exclude_mode: str | None = None,
):
    """
    Персональные рекомендации для пользователя (RetailRocket).
    Выводит item_id, category_id и краткое описание по умолчанию.

    exclude_mode:
      - None — взять из data/best_hyperparameters.json (поле recommend_filter) или 'all_train'
      - 'all_train' — не рекомендовать товары из train-истории
      - 'purchases_only' — скрывать только уже купленные в train
      - 'none' — не исключать просмотренные (filter_already_liked_items=False; вектор train всё равно нужен модели)
    """
    data, models = load_models()

    user_to_idx = data["user_to_idx"]
    idx_to_item = data["idx_to_item"]
    items_df = data["items"]

    if exclude_mode is None:
        rf = _load_recommend_filter_config()
        if rf in ("all_train", "purchases_only"):
            exclude_mode = rf
        else:
            exclude_mode = "all_train"

    user_id = int(user_id)
    if user_id not in user_to_idx:
        print(f"Пользователь с ID {user_id} не найден.")
        print(f"Примеры ID: {list(user_to_idx.keys())[:10]}")
        return None

    user_idx = user_to_idx[user_id]

    if model_name not in models:
        print(f"Модель {model_name} не загружена.")
        return None

    model = models[model_name]

    if exclude_mode == "purchases_only":
        user_items = load_npz(f"{DATA_DIR}/train_purchase_matrix.npz")[user_idx]
        filter_liked = True
    else:
        user_items = get_train_user_row(model_name, user_idx)
        filter_liked = exclude_mode != "none"

    recommended = model.recommend(
        user_idx,
        user_items,
        N=n_items,
        filter_already_liked_items=filter_liked,
    )
    recommended_idx = recommended[0]

    cat_map = {}
    for _, row in items_df.iterrows():
        iid = int(row["item_id"])
        cid = row["category_id"]
        cat_map[iid] = None if pd.isna(cid) else int(cid)

    print(f"\nРЕКОМЕНДАЦИИ ДЛЯ ПОЛЬЗОВАТЕЛЯ visitorid={user_id} (модель: {model_name}, фильтр: {exclude_mode})")
    print("=" * 90)

    recommendations = []
    for rank, item_idx in enumerate(recommended_idx, 1):
        item_id = idx_to_item.get(int(item_idx))
        if item_id is None:
            continue

        cid = cat_map.get(int(item_id))
        cat_str = "нет данных" if cid is None else str(cid)

        print(f"{rank:2d}. item_id={item_id}  |  category_id={cat_str}")
        print(f"    {DEFAULT_ITEM_DESCRIPTION}")
        print("-" * 90)

        recommendations.append(
            {
                "rank": rank,
                "item_id": int(item_id),
                "category_id": cid,
                "description": DEFAULT_ITEM_DESCRIPTION,
            }
        )

    rec_df = pd.DataFrame(recommendations)
    out = f"{DATA_DIR}/recommendations_user_{user_id}_{model_name}.csv"
    rec_df.to_csv(out, index=False, encoding="utf-8")
    print(f"\n[OK] Рекомендации сохранены: {out}")

    return rec_df


# ========== БАЗОВАЯ ЛИНИЯ ==========

def evaluate_baseline(train_binary, test_dict, test_users, k_values=(5, 10)):
    item_popularity = np.array(train_binary.sum(axis=0)).flatten()
    max_k = max(k_values)
    popular = np.argsort(item_popularity)[-max_k:][::-1].tolist()

    results = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}

    for user_idx in test_users:
        actual = test_dict.get(user_idx, [])
        if len(actual) == 0:
            continue
        for k in k_values:
            results[k]["precision"].append(precision_at_k(popular, actual, k))
            results[k]["recall"].append(recall_at_k(popular, actual, k))
            results[k]["ndcg"].append(ndcg_at_k(popular, actual, k))

    avg = {}
    for k in k_values:
        if len(results[k]["precision"]) == 0:
            avg[k] = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}
        else:
            avg[k] = {
                "precision": float(np.mean(results[k]["precision"])),
                "recall": float(np.mean(results[k]["recall"])),
                "ndcg": float(np.mean(results[k]["ndcg"])),
            }
    return avg


# ========== ГРАФИКИ ==========

def plot_model_comparison(
    results_path=f"{DATA_DIR}/model_comparison_results.csv",
    save_path=f"{DATA_DIR}/model_comparison.png",
):
    df = pd.read_csv(results_path)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    metrics = ["Precision@K", "Recall@K", "NDCG@K"]

    for i, metric in enumerate(metrics):
        sns.barplot(data=df, x="Модель", y=metric, hue="K", palette="viridis", ax=axes[i])
        axes[i].set_title(metric, fontsize=14, fontweight="bold")
        axes[i].set_ylabel(metric)
        axes[i].legend(title="K")
        for container in axes[i].containers:
            axes[i].bar_label(container, fmt="%.4f", fontsize=10)

    plt.suptitle("Сравнение качества рекомендательных моделей (RetailRocket)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[OK] График сравнения моделей сохранён: {save_path}")
    plt.show()


def plot_baseline_comparison(model_results, baseline_results, k=10, save_path=f"{DATA_DIR}/baseline_comparison.png"):
    rows = []
    for model_name, metrics in model_results.items():
        if k in metrics:
            for metric_name in ("precision", "recall", "ndcg"):
                rows.append(
                    {
                        "Модель": model_name,
                        "Метрика": f"{metric_name.capitalize()}@{k}",
                        "Значение": metrics[k][metric_name],
                    }
                )

    if k in baseline_results:
        for metric_name in ("precision", "recall", "ndcg"):
            rows.append(
                {
                    "Модель": "Popularity",
                    "Метрика": f"{metric_name.capitalize()}@{k}",
                    "Значение": baseline_results[k][metric_name],
                }
            )

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=df, x="Метрика", y="Значение", hue="Модель", palette="Set2", ax=ax)
    ax.set_title(f"Сравнение с базовой линией (K={k})", fontsize=16, fontweight="bold")
    ax.set_ylabel("Значение метрики")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.4f", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[OK] График сравнения с базовой линией сохранён: {save_path}")
    plt.show()


def print_baseline_table(model_results, baseline_results, k=10):
    if k not in baseline_results:
        print("Нет данных базовой линии для K =", k)
        return

    bl = baseline_results[k]

    best_name, best_p = None, -1.0
    for name, metrics in model_results.items():
        if k in metrics and metrics[k]["precision"] > best_p:
            best_p = metrics[k]["precision"]
            best_name = name

    if best_name is None:
        print("Нет данных моделей для K =", k)
        return

    bm = model_results[best_name][k]

    print(f"\nСравнение лучшей модели ({best_name}) с базовой линией (Popularity), K={k}:")
    print("-" * 65)
    print(f"{'Метрика':<16} {best_name:>12} {'Popularity':>12} {'Улучшение':>12}")
    print("-" * 65)
    for metric in ("precision", "recall", "ndcg"):
        mv = bm[metric]
        bv = bl[metric]
        if bv > 0:
            improvement = (mv - bv) / bv * 100
            imp_str = f"+{improvement:.1f}%"
        else:
            imp_str = "N/A"
        print(f"{metric.capitalize() + '@' + str(k):<16} {mv:>12.4f} {bv:>12.4f} {imp_str:>12}")
    print("-" * 65)


if __name__ == "__main__":
    print("=" * 90)
    print("ОЦЕНКА МОДЕЛЕЙ И СРАВНЕНИЕ С БАЗОВОЙ ЛИНИЕЙ (RetailRocket)")
    print("=" * 90)

    raw_data = load_data()
    _, train_binary, _, test_dict, test_users = train_test_split(raw_data, test_ratio=0.2)

    results_csv = pd.read_csv(f"{DATA_DIR}/model_comparison_results.csv")
    model_results = {}
    for _, row in results_csv.iterrows():
        name = row["Модель"]
        k = int(row["K"])
        if name not in model_results:
            model_results[name] = {}
        model_results[name][k] = {
            "precision": float(row["Precision@K"]),
            "recall": float(row["Recall@K"]),
            "ndcg": float(row["NDCG@K"]),
        }

    print("\nВычисление базовой линии (Popularity)...")
    baseline = evaluate_baseline(train_binary, test_dict, test_users, k_values=[5, 10])

    print("\nМетрики базовой линии (Popularity):")
    for k, m in sorted(baseline.items()):
        print(f"  K={k}: Precision={m['precision']:.4f}, Recall={m['recall']:.4f}, NDCG={m['ndcg']:.4f}")

    bl_rows = []
    for k, m in baseline.items():
        bl_rows.append(
            {
                "Модель": "Popularity",
                "K": k,
                "Precision@K": f"{m['precision']:.4f}",
                "Recall@K": f"{m['recall']:.4f}",
                "NDCG@K": f"{m['ndcg']:.4f}",
            }
        )
    bl_df = pd.DataFrame(bl_rows)
    bl_df.to_csv(f"{DATA_DIR}/baseline_results.csv", index=False)
    print("[OK] Базовая линия сохранена в data/baseline_results.csv")

    print_baseline_table(model_results, baseline, k=10)

    plot_model_comparison()
    plot_baseline_comparison(model_results, baseline, k=10)

    print("\n" + "=" * 90)
    print("ПРИМЕРЫ ПЕРСОНАЛЬНЫХ РЕКОМЕНДАЦИЙ")
    print("=" * 90)

    sample_uid = int(next(iter(raw_data["user_to_idx"].keys())))
    recommend_for_user(user_id=sample_uid, model_name="ALS", n_items=8, exclude_mode="all_train")
