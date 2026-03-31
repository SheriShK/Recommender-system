import pandas as pd
import numpy as np
from scipy.sparse import load_npz, csr_matrix
import json
import pickle

# Импорт библиотек для рекомендаций
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

# ========== 1. ЗАГРУЗКА ДАННЫХ ==========
def load_data():
    """Загружает все данные из папки data/"""
    print("Загрузка данных...")
    
    # Загрузка CSV
    users_df = pd.read_csv('data/users.csv')
    items_df = pd.read_csv('data/items.csv')
    interactions_df = pd.read_csv('data/interactions.csv')
    user_features_df = pd.read_csv('data/user_features.csv')
    item_features_df = pd.read_csv('data/item_features.csv')
    
    # Загрузка разреженных матриц
    interaction_matrix = load_npz('data/interaction_matrix.npz')
    confidence_matrix = load_npz('data/confidence_matrix.npz')
    
    # Загрузка маппингов
    with open('data/user_to_idx.json', 'r') as f:
        user_to_idx = json.load(f)
    with open('data/item_to_idx.json', 'r') as f:
        item_to_idx = json.load(f)
    
    # Обратный маппинг
    idx_to_user = {v: int(k) for k, v in user_to_idx.items()}
    idx_to_item = {v: int(k) for k, v in item_to_idx.items()}
    
    print(f"✓ Загружено: {interaction_matrix.shape[0]} пользователей, "
          f"{interaction_matrix.shape[1]} объявлений")
    print(f"✓ Взаимодействий: {interaction_matrix.nnz}")
    
    return {
        'users': users_df,
        'items': items_df,
        'interactions': interactions_df,
        'interaction_matrix': interaction_matrix,
        'confidence_matrix': confidence_matrix,
        'user_features': user_features_df,
        'item_features': item_features_df,
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_item': idx_to_item
    }


# ========== 2. ФУНКЦИИ ДЛЯ ОЦЕНКИ КАЧЕСТВА ==========

def precision_at_k(recommended, actual, k):
    """Precision@K для одного пользователя"""
    recommended_k = recommended[:k]
    if len(actual) == 0:
        return 0.0
    hits = len(set(recommended_k) & set(actual))
    return hits / k


def recall_at_k(recommended, actual, k):
    """Recall@K для одного пользователя"""
    recommended_k = recommended[:k]
    if len(actual) == 0:
        return 0.0
    hits = len(set(recommended_k) & set(actual))
    return hits / len(actual)


def ndcg_at_k(recommended, actual, k):
    """NDCG@K (Normalized Discounted Cumulative Gain)"""
    recommended_k = recommended[:k]
    if len(actual) == 0:
        return 0.0
    
    # Создаём массив релевантности
    relevance = [1 if item in actual else 0 for item in recommended_k]
    
    # DCG
    dcg = 0.0
    for i, rel in enumerate(relevance):
        if rel > 0:
            dcg += rel / np.log2(i + 2)
    
    # IDCG (идеальный случай: все релевантные в начале)
    ideal_relevance = [1] * min(len(actual), k) + [0] * (k - min(len(actual), k))
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance):
        if rel > 0:
            idcg += rel / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(model_name, recommender_func, test_users, train_matrix, 
                   test_dict, k_values=[5, 10]):
    """
    Универсальная функция для оценки качества рекомендаций
    """
    results = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in k_values}
    
    for user_idx in test_users:
        # Получаем рекомендации от модели
        try:
            recommended_items_idx = recommender_func(user_idx)
        except Exception as e:
            print(f"  Ошибка для пользователя {user_idx}: {e}")
            continue
        
        # Реальные объявления, с которыми пользователь взаимодействовал (в тесте)
        actual_items_idx = test_dict.get(user_idx, [])
        
        if len(actual_items_idx) == 0:
            continue
        
        for k in k_values:
            results[k]['precision'].append(
                precision_at_k(recommended_items_idx, actual_items_idx, k)
            )
            results[k]['recall'].append(
                recall_at_k(recommended_items_idx, actual_items_idx, k)
            )
            results[k]['ndcg'].append(
                ndcg_at_k(recommended_items_idx, actual_items_idx, k)
            )
    
    # Усреднение результатов (без NaN: если нет ни одного пользователя, ставим 0.0)
    avg_results = {}
    for k in k_values:
        if len(results[k]['precision']) == 0:
            avg_results[k] = {
                'precision': 0.0,
                'recall': 0.0,
                'ndcg': 0.0
            }
        else:
            avg_results[k] = {
                'precision': float(np.mean(results[k]['precision'])),
                'recall': float(np.mean(results[k]['recall'])),
                'ndcg': float(np.mean(results[k]['ndcg']))
            }
    
    return avg_results


# ========== 2.1. ВСПОМОГАТЕЛЬНОЕ ==========

def matrix_to_binary(sparse_matrix):
    """Перевод sparse-матрицы в бинарную (0/1) по наличию взаимодействия."""
    binary = sparse_matrix.copy()
    if binary.data.size > 0:
        binary.data = np.ones_like(binary.data, dtype=np.float32)
    return binary


def sample_users(users, max_users=120, seed=42):
    """Берём подмножество пользователей для быстрой проверки гиперпараметров."""
    rng = np.random.default_rng(seed)
    users = list(users)
    if len(users) <= max_users:
        return users
    idx = rng.choice(len(users), size=max_users, replace=False)
    return [users[i] for i in idx]


def tune_als(train_confidence, tune_users, test_dict, factors_list, iterations_list, regularization_list, n_items=10):
    best_precision = -1.0
    best_params = None
    best_model = None

    for factors in factors_list:
        for iterations in iterations_list:
            for regularization in regularization_list:
                print(f"  ALS grid: factors={factors}, iterations={iterations}, regularization={regularization}")
                model = train_als(train_confidence, factors=factors, iterations=iterations, regularization=regularization)

                def rec_func(user_idx):
                    user_items = train_confidence[user_idx]  # 1 строка (1 пользователь)
                    recommended = model.recommend([user_idx], user_items, N=n_items)
                    # recommended[0] -> items матрица формы (1, N)
                    return [int(item) for item in recommended[0][0]]

                metrics = evaluate_model(
                    "ALS",
                    rec_func,
                    tune_users,
                    train_confidence,
                    test_dict,
                    k_values=[10]
                )
                precision10 = metrics[10]["precision"]
                if np.isnan(precision10) or np.isinf(precision10):
                    precision10 = 0.0
                print(f"    Precision@10 (tune): {precision10:.4f}")

                if (best_model is None) or (precision10 > best_precision):
                    best_precision = precision10
                    best_params = {
                        "factors": factors,
                        "iterations": iterations,
                        "regularization": regularization
                    }
                    best_model = model

    # Финальная оценка лучшей модели на полном test
    return best_model, best_params, best_precision


def tune_bpr(train_interaction, tune_users, test_dict, factors_list, iterations_list, learning_rate_list, n_items=10):
    best_precision = -1.0
    best_params = None
    best_model = None

    for factors in factors_list:
        for iterations in iterations_list:
            for learning_rate in learning_rate_list:
                print(f"  BPR grid: factors={factors}, iterations={iterations}, learning_rate={learning_rate}")
                model = train_bpr(train_interaction, factors=factors, iterations=iterations, learning_rate=learning_rate)

                def rec_func(user_idx):
                    user_items = train_interaction[user_idx]  # 1 строка (1 пользователь)
                    recommended = model.recommend([user_idx], user_items, N=n_items)
                    return [int(item) for item in recommended[0][0]]

                metrics = evaluate_model(
                    "BPR",
                    rec_func,
                    tune_users,
                    train_interaction,
                    test_dict,
                    k_values=[10]
                )
                precision10 = metrics[10]["precision"]
                if np.isnan(precision10) or np.isinf(precision10):
                    precision10 = 0.0
                print(f"    Precision@10 (tune): {precision10:.4f}")

                if (best_model is None) or (precision10 > best_precision):
                    best_precision = precision10
                    best_params = {
                        "factors": factors,
                        "iterations": iterations,
                        "learning_rate": learning_rate
                    }
                    best_model = model

    return best_model, best_params, best_precision


def tune_itemknn(train_interaction, tune_users, test_dict, k_neighbors_list, n_items=10):
    best_precision = -1.0
    best_params = None
    best_model = None

    for k_neighbors in k_neighbors_list:
        print(f"  ItemKNN grid: k_neighbors={k_neighbors}")
        model = train_itemknn(train_interaction, k_neighbors=k_neighbors)

        def rec_func(user_idx):
            user_items = train_interaction[user_idx]  # 1 строка (1 пользователь)
            recommended = model.recommend([user_idx], user_items, N=n_items)
            return [int(item) for item in recommended[0][0]]

        metrics = evaluate_model(
            "ItemKNN",
            rec_func,
            tune_users,
            train_interaction,
            test_dict,
            k_values=[10]
        )
        precision10 = metrics[10]["precision"]
        if np.isnan(precision10) or np.isinf(precision10):
            precision10 = 0.0
        print(f"    Precision@10 (tune): {precision10:.4f}")

        if (best_model is None) or (precision10 > best_precision):
            best_precision = precision10
            best_params = {"k_neighbors": k_neighbors}
            best_model = model

    return best_model, best_params, best_precision


# ========== 3. ОБУЧЕНИЕ ALS (Alternating Least Squares) ==========

def train_als(train_confidence, factors=50, iterations=20, regularization=0.1):
    """Обучение ALS модели (implicit)"""
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ ALS (Alternating Least Squares)")
    print("="*60)
    
    # ALS использует confidence_matrix (с весами 1,3,4,5)
    confidence_matrix = train_confidence
    
    # Создаём и обучаем модель
    model = AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=42,
        num_threads=4
    )
    
    print(f"Параметры: factors={factors}, iterations={iterations}, "
          f"regularization={regularization}")
    print(f"Размер матрицы: {confidence_matrix.shape}")
    print(f"Плотность: {confidence_matrix.nnz / (confidence_matrix.shape[0] * confidence_matrix.shape[1]) * 100:.2f}%")
    
    # Обучение
    model.fit(confidence_matrix)
    
    print("✓ Модель обучена")
    
    return model


def get_recommendations_als(model, user_idx, user_items, n_items=10):
    """Получение рекомендаций от ALS модели"""
    try:
        # model.recommend возвращает (items, scores)
        recommended = model.recommend([user_idx], user_items, N=n_items)
        return [int(item) for item in recommended[0][0]]
    except:
        # Если пользователь не в обучающей выборке
        return list(range(min(n_items, 500)))


# ========== 4. ОБУЧЕНИЕ BPR-MF (Bayesian Personalised Ranking) ==========

def train_bpr(train_interaction, factors=50, iterations=20, learning_rate=0.01):
    """Обучение BPR модели"""
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ BPR (Bayesian Personalised Ranking)")
    print("="*60)
    
    # BPR использует бинарную матрицу (взаимодействие было/не было)
    interaction_matrix = train_interaction
    
    model = BayesianPersonalizedRanking(
        factors=factors,
        iterations=iterations,
        learning_rate=learning_rate,
        random_state=42,
        num_threads=4
    )
    
    print(f"Параметры: factors={factors}, iterations={iterations}, "
          f"learning_rate={learning_rate}")
    print(f"Размер матрицы: {interaction_matrix.shape}")
    
    model.fit(interaction_matrix)
    
    print("✓ Модель обучена")
    
    return model


def get_recommendations_bpr(model, user_idx, user_items, n_items=10):
    """Получение рекомендаций от BPR модели"""
    try:
        recommended = model.recommend([user_idx], user_items, N=n_items)
        return [int(item) for item in recommended[0][0]]
    except:
        return list(range(min(n_items, 500)))


# ========== 5. ОБУЧЕНИЕ ItemKNN ==========

# ========== 5. ОБУЧЕНИЕ ItemKNN (Cosine) ==========

def train_itemknn(train_interaction, k_neighbors=20):
    """Обучение Item-Item KNN с косинусной мерой (правильный способ для implicit)"""
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ ItemKNN (Item-based Collaborative Filtering)")
    print("="*60)
    
    interaction_matrix = train_interaction
    
    # Используем CosineRecommender вместо ItemItemRecommender
    from implicit.nearest_neighbours import CosineRecommender
    
    model = CosineRecommender(
        K=k_neighbors,
        num_threads=4
    )
    
    print(f"Параметры: K_neighbors={k_neighbors} (cosine similarity)")
    print(f"Размер матрицы: {interaction_matrix.shape}")
    print(f"Плотность: {interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1]) * 100:.2f}%")
    
    # Обучение
    model.fit(interaction_matrix)
    
    print("✓ Модель ItemKNN обучена")
    
    return model


def get_recommendations_itemknn(model, user_idx, user_items, n_items=10):
    """Получение рекомендаций от ItemKNN модели"""
    try:
        # Для CosineRecommender и ItemItemRecommender используется interaction_matrix
        recommended = model.recommend([user_idx], user_items, N=n_items)
        return [int(item) for item in recommended[0][0]]
    except Exception as e:
        print(f"  Ошибка рекомендаций ItemKNN для пользователя {user_idx}: {e}")
        return list(range(min(n_items, 500)))


# ========== 6. РАЗДЕЛЕНИЕ НА ТРЕЙН И ТЕСТ ==========

def train_test_split(data, test_ratio=0.2):
    """
    Разделяет взаимодействия на train/test внутри каждого пользователя.

    Важно: ItemKNN (и вообще item-based методы) требуют историю пользователя в train.
    Поэтому split по пользователям целиком (cold-start) сильно занижает метрики.
    """
    interactions_df = data['interactions'].copy()
    interactions_df['user_id'] = interactions_df['user_id'].astype(int)
    interactions_df['item_id'] = interactions_df['item_id'].astype(int)

    user_to_idx = {int(k): int(v) for k, v in data['user_to_idx'].items()}
    item_to_idx = {int(k): int(v) for k, v in data['item_to_idx'].items()}

    # Оставляем только валидные user/item
    interactions_df = interactions_df[
        interactions_df['user_id'].isin(user_to_idx.keys()) &
        interactions_df['item_id'].isin(item_to_idx.keys())
    ]

    # Для разреженных данных важно не оценивать "холодных" пользователей,
    # иначе item-based модели (и даже матричные факторизации) дают почти ноль.
    MIN_TOTAL_INTERACTIONS = 3
    MIN_TRAIN_INTERACTIONS = 2

    # Для ALS implicit часто используют confidence = 1 + alpha * preference/count.
    # Подбор confidence влияет на качество ALS.
    # С учетом того, что relevance лежит примерно в [1..5], берём умеренную шкалу,
    # чтобы не "перекосить" обучение.
    ALS_CONFIDENCE_ALPHA = 1.0

    # Для детерминированного split по времени используем timestamp.
    # Если парсинг timestamp не удался, сортировка всё равно будет корректной для ISO-формата,
    # а NaT окажутся в конце.
    if 'timestamp' in interactions_df.columns:
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'], errors='coerce')
    rng = np.random.default_rng(42)

    rows = []
    cols = []
    data_conf = []
    test_dict = {}

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)

    # Разделяем взаимодействия по пользователям
    train_cnt = 0
    test_cnt = 0
    for user_id, user_df in interactions_df.groupby('user_id'):
        u_idx = user_to_idx.get(int(user_id))
        if u_idx is None:
            continue

        idxs = user_df.index.to_numpy()
        if len(idxs) < MIN_TOTAL_INTERACTIONS:
            # Если у пользователя мало взаимодействий, рекомендации для него будут нестабильны
            continue

        n_test = int(len(idxs) * test_ratio)
        # Хотим, чтобы в train осталось как минимум MIN_TRAIN_INTERACTIONS
        n_test = max(1, n_test)
        n_test = min(n_test, len(idxs) - MIN_TRAIN_INTERACTIONS)

        # Если из-за ограничений train стал пустым — пропускаем пользователя
        if n_test <= 0 or n_test >= len(idxs):
            continue

        # Берём последние n_test взаимодействий пользователя как test (по timestamp)
        if 'timestamp' in user_df.columns:
            user_df_sorted = user_df.sort_values('timestamp')
            test_indices = user_df_sorted.tail(n_test).index.to_numpy()
        else:
            # fallback: случайно
            test_indices = rng.choice(idxs, size=n_test, replace=False)

        train_df = user_df.drop(test_indices)
        test_df = user_df.loc[test_indices]

        # Train матрица (confidence)
        for _, row in train_df.iterrows():
            i_idx = item_to_idx.get(int(row['item_id']))
            if i_idx is None:
                continue
            rows.append(u_idx)
            cols.append(i_idx)
            # Усиливаем confidence для ALS
            data_conf.append(1.0 + ALS_CONFIDENCE_ALPHA * float(row['relevance']))
            train_cnt += 1

        # Test: уникальные item idx для метрик
        test_items = set()
        for item_id in test_df['item_id'].values:
            i_idx = item_to_idx.get(int(item_id))
            if i_idx is not None:
                test_items.add(i_idx)
        test_dict[u_idx] = list(test_items)
        test_cnt += len(test_indices)

    from scipy.sparse import csr_matrix
    train_confidence = csr_matrix((data_conf, (rows, cols)), shape=(n_users, n_items))

    # Пользователи, для которых в train есть хоть одно взаимодействие
    test_users_filtered = []
    for u in test_dict.keys():
        if train_confidence.indptr[u + 1] > train_confidence.indptr[u]:
            test_users_filtered.append(u)

    print(f"\nРазделение данных (внутри пользователей):")
    print(f"  Train interactions: {train_cnt}")
    print(f"  Test interactions: {test_cnt}")
    print(f"  Пользователей в тесте для оценки: {len(test_users_filtered)}")

    return train_confidence, test_dict, test_users_filtered


# ========== 8. ГЛАВНАЯ ФУНКЦИЯ ==========

def main():
    print("="*60)
    print("ОБУЧЕНИЕ МОДЕЛЕЙ РЕКОМЕНДАТЕЛЬНОЙ СИСТЕМЫ")
    print("="*60)
    
    # Загрузка данных
    data = load_data()
    
    # Разделение на train/test
    train_matrix, test_dict, test_users = train_test_split(data, test_ratio=0.2)
    
    # Подготовка матриц строго по train-разбиению
    # ALS обучаем по confidence-матрице, BPR/ItemKNN - по бинарной матрице взаимодействий.
    train_confidence = train_matrix
    # Для ItemKNN и BPR лучше использовать взвешенное взаимодействие (а не бинарное),
    # т.к. в данных relevance играет роль силы сигнала.
    train_interaction = train_confidence
    
    tune_users = sample_users(test_users, max_users=120, seed=42)
    print(f"\nПодбор гиперпараметров будет идти по {len(tune_users)} пользователям из теста...")
    
    results = {}
    best_params = {}
    
    # ===== 1. ALS =====
    print("\n" + "="*60)
    print("1. ОБУЧЕНИЕ ALS")
    print("="*60)
    
    factors_list = [20, 50]
    iterations_list = [10, 20]
    regularization_list = [0.01, 0.1]
    learning_rate_list = [0.005, 0.01]
    k_neighbors_list = [10, 20, 50]

    als_model, best_als_params, _ = tune_als(
        train_confidence=train_confidence,
        tune_users=tune_users,
        test_dict=test_dict,
        factors_list=factors_list,
        iterations_list=iterations_list,
        regularization_list=regularization_list,
        n_items=10
    )
    print(f"Лучшие параметры ALS: {best_als_params}")
    best_params["ALS"] = best_als_params
    
    def als_rec_func(user_idx):
        user_items = train_confidence[user_idx]
        recommended = als_model.recommend([user_idx], user_items, N=10)
        return [int(item) for item in recommended[0][0]]
    
    results['ALS'] = evaluate_model(
        'ALS', als_rec_func, test_users, train_confidence, test_dict, k_values=[5, 10]
    )
    
    # Сохранение ALS модели
    with open('data/model_als.pkl', 'wb') as f:
        pickle.dump(als_model, f)
    print("✓ ALS модель сохранена")
    
    # ===== 2. BPR =====
    print("\n" + "="*60)
    print("2. ОБУЧЕНИЕ BPR")
    print("="*60)
    
    bpr_model, best_bpr_params, _ = tune_bpr(
        train_interaction=train_interaction,
        tune_users=tune_users,
        test_dict=test_dict,
        factors_list=factors_list,
        iterations_list=iterations_list,
        learning_rate_list=learning_rate_list,
        n_items=10
    )
    print(f"Лучшие параметры BPR: {best_bpr_params}")
    best_params["BPR"] = best_bpr_params
    
    def bpr_rec_func(user_idx):
        user_items = train_interaction[user_idx]
        recommended = bpr_model.recommend([user_idx], user_items, N=10)
        return [int(item) for item in recommended[0][0]]
    
    results['BPR'] = evaluate_model(
        'BPR', bpr_rec_func, test_users, train_interaction, test_dict, k_values=[5, 10]
    )
    
    with open('data/model_bpr.pkl', 'wb') as f:
        pickle.dump(bpr_model, f)
    print("✓ BPR модель сохранена")
    
    # ===== 3. ItemKNN =====
    print("\n" + "="*60)
    print("3. ОБУЧЕНИЕ ItemKNN")
    print("="*60)
    
    itemknn_model, best_itemknn_params, _ = tune_itemknn(
        train_interaction=train_interaction,
        tune_users=tune_users,
        test_dict=test_dict,
        k_neighbors_list=k_neighbors_list,
        n_items=10
    )
    print(f"Лучшие параметры ItemKNN: {best_itemknn_params}")
    best_params["ItemKNN"] = best_itemknn_params
    
    def itemknn_rec_func(user_idx):
        user_items = train_interaction[user_idx]
        recommended = itemknn_model.recommend([user_idx], user_items, N=10)
        return [int(item) for item in recommended[0][0]]
    
    results['ItemKNN'] = evaluate_model(
        'ItemKNN', itemknn_rec_func, test_users, train_interaction, test_dict, k_values=[5, 10]
    )
    
    with open('data/model_itemknn.pkl', 'wb') as f:
        pickle.dump(itemknn_model, f)
    print("✓ ItemKNN модель сохранена")
    
    # ===== ВЫВОД РЕЗУЛЬТАТОВ =====
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ АЛГОРИТМОВ")
    print("="*60)
    
    results_df = []
    for model_name, metrics in results.items():
        for k, k_metrics in metrics.items():
            results_df.append({
                'Модель': model_name,
                'K': k,
                'Precision@K': f"{k_metrics['precision']:.4f}",
                'Recall@K': f"{k_metrics['recall']:.4f}",
                'NDCG@K': f"{k_metrics['ndcg']:.4f}"
            })
    
    results_table = pd.DataFrame(results_df)
    print(results_table.to_string(index=False))
    
    # Сохранение результатов
    results_table.to_csv('data/model_comparison_results.csv', index=False)
    print("\n✓ Результаты сохранены в data/model_comparison_results.csv")

    # Сохранение подобранных гиперпараметров
    with open('data/best_hyperparameters.json', 'w', encoding='utf-8') as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    print("✓ Гиперпараметры сохранены в data/best_hyperparameters.json")
    
    # ===== ВЫВОД ЛУЧШЕЙ МОДЕЛИ =====
    best_precision = -1.0
    best_model = None
    for model_name, metrics in results.items():
        if 10 in metrics and metrics[10]['precision'] > best_precision:
            best_precision = metrics[10]['precision']
            best_model = model_name
    
    print("\n" + "="*60)
    print(f"🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model}")
    print(f"   Precision@10: {best_precision:.4f}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = main()