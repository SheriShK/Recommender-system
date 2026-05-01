"""
Подготовка данных RetailRocket Kaggle: events + item_properties + category_tree.
Создаёт data/items.csv, data/interactions.csv, маппинги и sparse-матрицы (полная история).
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz

# ========== Пути ==========
SCRIPT_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = SCRIPT_DIR / "archive"
DATA_DIR = SCRIPT_DIR / "data"

EVENT_WEIGHTS = {
    "view": 1,
    "addtocart": 3,
    "transaction": 5,
}

ITEM_PROPERTY_FILES = ("item_properties_part1.csv", "item_properties_part2.csv")

# Минимум взаимодействий на пользователя после фильтрации
MIN_INTERACTIONS_PER_USER = 2


def parse_property_value(raw) -> float | None:
    """Число из value: целое/вещественное или префикс 'n' (RetailRocket)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s.startswith("n"):
        tail = s[1:].split()[0]
        try:
            return float(tail)
        except ValueError:
            return None
    try:
        return float(s.split()[0])
    except ValueError:
        return None


def load_item_properties_latest(archive: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Для каждого (itemid, property) берётся последняя по timestamp запись.
    Возвращает таблицы для property == 'categoryid' и 'available' (по одной строке на item).
    """
    chunks_cat = []
    chunks_av = []
    usecols = ["timestamp", "itemid", "property", "value"]

    for name in ITEM_PROPERTY_FILES:
        path = archive / name
        if not path.exists():
            continue
        for ch in pd.read_csv(path, usecols=usecols, chunksize=400_000, low_memory=False):
            ch = ch[ch["property"].astype(str).isin(("categoryid", "available"))]
            if ch.empty:
                continue
            c_id = ch[ch["property"].astype(str) == "categoryid"]
            c_av = ch[ch["property"].astype(str) == "available"]
            if not c_id.empty:
                chunks_cat.append(c_id)
            if not c_av.empty:
                chunks_av.append(c_av)

    if not chunks_cat and not chunks_av:
        raise FileNotFoundError(
            f"Не найдены {ITEM_PROPERTY_FILES} или нет строк categoryid/available в {archive}"
        )

    cat_df = pd.concat(chunks_cat, ignore_index=True) if chunks_cat else pd.DataFrame()
    av_df = pd.concat(chunks_av, ignore_index=True) if chunks_av else pd.DataFrame()

    for df in (cat_df, av_df):
        if df.empty:
            continue
        df.sort_values("timestamp", inplace=True)
    if not cat_df.empty:
        cat_df = cat_df.drop_duplicates(subset=["itemid", "property"], keep="last")
    if not av_df.empty:
        av_df = av_df.drop_duplicates(subset=["itemid", "property"], keep="last")

    return cat_df, av_df


def build_category_map(cat_df: pd.DataFrame) -> dict[int, int]:
    out: dict[int, int] = {}
    if cat_df.empty:
        return out
    for _, row in cat_df.iterrows():
        iid = int(row["itemid"])
        v = parse_property_value(row["value"])
        if v is not None and not np.isnan(v):
            out[iid] = int(v)
    return out


def load_events(archive: Path) -> pd.DataFrame:
    path = archive / "events.csv"
    if not path.exists():
        raise FileNotFoundError(f"Нет файла {path}")
    ev = pd.read_csv(path, dtype={"visitorid": np.int64, "itemid": np.int64})
    ev["event"] = ev["event"].astype(str).str.strip().str.lower()
    ok = ev["event"].isin(EVENT_WEIGHTS.keys())
    ev = ev.loc[ok].copy()
    ev["relevance"] = ev["event"].map(EVENT_WEIGHTS)
    ev["timestamp"] = pd.to_numeric(ev["timestamp"], errors="coerce").astype("Int64")
    ev = ev.dropna(subset=["timestamp"])
    ev["timestamp"] = ev["timestamp"].astype(np.int64)
    return ev


def filter_users_min_events(ev: pd.DataFrame, min_count: int) -> pd.DataFrame:
    cnt = ev.groupby("visitorid").size()
    keep = cnt[cnt >= min_count].index
    return ev[ev["visitorid"].isin(keep)].copy()


def build_indices(ev: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    u_ids = np.sort(ev["visitorid"].unique())
    i_ids = np.sort(ev["itemid"].unique())
    user_to_idx = {int(u): i for i, u in enumerate(u_ids)}
    item_to_idx = {int(i): j for j, i in enumerate(i_ids)}
    return user_to_idx, item_to_idx


def aggregate_matrix_rows(
    ev: pd.DataFrame,
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
) -> tuple[list[int], list[int], list[float], list[float]]:
    """Агрегация (user, item): max relevance для confidence; binary = 1 при наличии события."""
    ev = ev.copy()
    ev["u"] = ev["visitorid"].map(user_to_idx)
    ev["i"] = ev["itemid"].map(item_to_idx)
    ev = ev.dropna(subset=["u", "i"])
    ev["u"] = ev["u"].astype(np.int32)
    ev["i"] = ev["i"].astype(np.int32)

    g = ev.groupby(["u", "i"], sort=False)
    max_rel = g["relevance"].max()
    rows = [int(a) for a, _ in max_rel.index]
    cols = [int(b) for _, b in max_rel.index]
    conf_data = max_rel.values.astype(np.float64).tolist()
    bin_data = [1.0] * len(rows)
    return rows, cols, conf_data, bin_data


def copy_category_tree(archive: Path, data_dir: Path) -> None:
    src = archive / "category_tree.csv"
    if src.exists():
        import shutil

        shutil.copy2(src, data_dir / "category_tree.csv")


def main():
    print("RetailRocket: подготовка данных...")
    os.makedirs(DATA_DIR, exist_ok=True)

    copy_category_tree(ARCHIVE_DIR, DATA_DIR)

    print("  Загрузка item_properties (последние значения categoryid / available)...")
    cat_df, av_df = load_item_properties_latest(ARCHIVE_DIR)
    category_by_item = build_category_map(cat_df)
    print(f"    Категории для {len(category_by_item)} товаров")

    print("  Загрузка events.csv...")
    ev = load_events(ARCHIVE_DIR)
    print(f"    Строк событий (известные типы): {len(ev)}")

    print(f"  Фильтр: пользователи с >= {MIN_INTERACTIONS_PER_USER} взаимодействиями...")
    ev = filter_users_min_events(ev, MIN_INTERACTIONS_PER_USER)
    print(f"    Строк после фильтра: {len(ev)}")

    user_to_idx, item_to_idx = build_indices(ev)
    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    print(f"    Пользователей: {n_users}, товаров: {n_items}")

    # items.csv: все item_id из матрицы + category_id
    item_ids_sorted = sorted(item_to_idx.keys())
    items_rows = []
    for iid in item_ids_sorted:
        cid = category_by_item.get(int(iid), np.nan)
        items_rows.append({"item_id": int(iid), "category_id": cid})
    items_df = pd.DataFrame(items_rows)
    items_df.to_csv(DATA_DIR / "items.csv", index=False)

    # Метаданные: явный JSON со списком item -> category
    meta_items = {int(r["item_id"]): (None if pd.isna(r["category_id"]) else int(r["category_id"])) for _, r in items_df.iterrows()}
    with open(DATA_DIR / "items_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {"source": "RetailRocket", "n_items": len(meta_items), "item_category": meta_items},
            f,
            ensure_ascii=False,
        )

    interactions_out = ev.rename(columns={"visitorid": "user_id", "itemid": "item_id"})[
        ["user_id", "item_id", "event", "relevance", "timestamp"]
    ].copy()
    interactions_out = interactions_out.rename(columns={"event": "action_type"})
    interactions_out.to_csv(DATA_DIR / "interactions.csv", index=False)

    rows, cols, conf_data, bin_data = aggregate_matrix_rows(ev, user_to_idx, item_to_idx)
    interaction_matrix = csr_matrix((bin_data, (rows, cols)), shape=(n_users, n_items))
    confidence_matrix = csr_matrix((conf_data, (rows, cols)), shape=(n_users, n_items))

    nnz = interaction_matrix.nnz
    dens = nnz / max(1, (n_users * n_items))
    print("\n" + "=" * 50)
    print("СТАТИСТИКА МАТРИЦЫ (после агрегации max по паре user-item):")
    print(f"  nnz: {nnz}")
    print(f"  Плотность: {dens * 100:.4f}% (~{(1 - dens) * 100:.2f}% нулей)")
    print(f"  Среднее взаимодействий на пользователя: {nnz / max(1, n_users):.2f}")

    save_npz(DATA_DIR / "interaction_matrix.npz", interaction_matrix)
    save_npz(DATA_DIR / "confidence_matrix.npz", confidence_matrix)

    with open(DATA_DIR / "user_to_idx.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in user_to_idx.items()}, f)
    with open(DATA_DIR / "item_to_idx.json", "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in item_to_idx.items()}, f)

    with open(DATA_DIR / "user_to_idx.pkl", "wb") as f:
        pickle.dump(user_to_idx, f)
    with open(DATA_DIR / "item_to_idx.pkl", "wb") as f:
        pickle.dump(item_to_idx, f)

    # available: сохраним опционально как словарь (не users.csv / не item_features)
    available_map: dict[int, int] = {}
    if not av_df.empty:
        for _, row in av_df.iterrows():
            iid = int(row["itemid"])
            v = parse_property_value(row["value"])
            if v is not None:
                available_map[iid] = int(v) if v in (0, 1) else int(round(v))
        with open(DATA_DIR / "item_available.json", "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in available_map.items()}, f)

    print("\n[OK] Сохранено в папку data/: items.csv, interactions.csv, *_matrix.npz, маппинги, items_metadata.json")
    if available_map:
        print("  Дополнительно: item_available.json (0/1 по последнему timestamp)")

    return {
        "events_rows": len(ev),
        "n_users": n_users,
        "n_items": n_items,
        "density_pct": float(dens * 100),
    }


if __name__ == "__main__":
    main()
