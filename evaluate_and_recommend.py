# ================================================
# evaluate_and_recommend.py
# Исправленная версия для ВКР (Пункты A + C)
# ================================================

import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.sparse import load_npz

# Настройки графиков
sns.set(style="whitegrid", font_scale=1.2)

def load_all_data():
    """Загружает данные и модели"""
    print("Загрузка данных и моделей...")
    
    # Загружаем основные данные
    users_df = pd.read_csv('data/users.csv')
    items_df = pd.read_csv('data/items.csv')
    
    with open('data/user_to_idx.json', 'r', encoding='utf-8') as f:
        user_to_idx = json.load(f)
    with open('data/item_to_idx.json', 'r', encoding='utf-8') as f:
        item_to_idx = json.load(f)
    
    # Приводим ключи к int
    user_to_idx = {int(k): int(v) for k, v in user_to_idx.items()}
    idx_to_user = {v: int(k) for k, v in user_to_idx.items()}
    idx_to_item = {v: int(k) for k, v in item_to_idx.items()}
    
    # Загружаем модели
    models = {}
    model_files = {
        'ALS': 'data/model_als.pkl',
        'BPR': 'data/model_bpr.pkl',
        'ItemKNN': 'data/model_itemknn.pkl'
    }
    
    for name, path in model_files.items():
        try:
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
        except FileNotFoundError:
            print(f"⚠️ Модель {name} не найдена")
    
    print(f"✓ Загружено {len(models)} моделей")
    print(f"✓ Всего пользователей в данных: {len(user_to_idx)}")
    
    return {
        'users': users_df,
        'items': items_df,
        'user_to_idx': user_to_idx,
        'idx_to_item': idx_to_item,
        'idx_to_user': idx_to_user
    }, models


def plot_model_comparison(results_path='data/model_comparison_results.csv', 
                          save_path='data/model_comparison.png'):
    """Красивый график сравнения моделей"""
    df = pd.read_csv(results_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    metrics = ['Precision@K', 'Recall@K', 'NDCG@K']
    
    for i, metric in enumerate(metrics):
        sns.barplot(data=df, x='Модель', y=metric, hue='K', palette='viridis', ax=axes[i])
        axes[i].set_title(metric, fontsize=14, fontweight='bold')
        axes[i].set_ylabel(metric)
        axes[i].legend(title='K')
        
        # Значения над столбцами
        for container in axes[i].containers:
            axes[i].bar_label(container, fmt='%.4f', fontsize=10)
    
    plt.suptitle('Сравнение качества рекомендательных моделей', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ График успешно сохранён: {save_path}")
    plt.show()


def recommend_for_user(user_id: int, model_name: str = 'ALS', n_items: int = 10):
    """Персональные рекомендации для пользователя"""
    data, models = load_all_data()
    
    user_to_idx = data['user_to_idx']
    idx_to_item = data['idx_to_item']
    items_df = data['items']
    
    user_id = int(user_id)
    
    if user_id not in user_to_idx:
        print(f"❌ Пользователь с ID {user_id} не найден в данных.")
        # Покажем несколько существующих пользователей
        print(f"Доступные ID пользователей (первые 10): {list(user_to_idx.keys())[:10]}")
        return None
    
    user_idx = user_to_idx[user_id]
    
    if model_name not in models:
        print(f"❌ Модель {model_name} не загружена.")
        return None
    
    model = models[model_name]
    
    # Получаем матрицу для пользователя
    if model_name == 'ALS':
        matrix = load_npz('data/confidence_matrix.npz')
    else:
        matrix = load_npz('data/interaction_matrix.npz')
    
    user_items = matrix[user_idx]
    
    # Получаем рекомендации
    recommended = model.recommend([user_idx], user_items, N=n_items)
    recommended_idx = recommended[0][0]   # берем индексы
    
    print(f"\n🎯 РЕКОМЕНДАЦИИ ДЛЯ ПОЛЬЗОВАТЕЛЯ ID = {user_id} (модель: {model_name})")
    print("=" * 95)
    
    recommendations = []
    for rank, item_idx in enumerate(recommended_idx, 1):
        item_id = idx_to_item.get(int(item_idx))
        if item_id is None:
            continue
            
        item = items_df[items_df['item_id'] == item_id].iloc[0]
        
        print(f"{rank:2d}. [{item_id:3d}] {item['title'][:70]:70}...")
        print(f"     → Категория: {item['category_name']:20} | "
              f"Цена: {item['price']:,.0f} ₽ | Город: {item['city']}")
        print("-" * 90)
        
        recommendations.append({
            'rank': rank,
            'item_id': int(item_id),
            'title': item['title'],
            'category': item['category_name'],
            'price': float(item['price']),
            'city': item['city']
        })
    
    # Сохраняем в файл
    rec_df = pd.DataFrame(recommendations)
    rec_df.to_csv(f'data/recommendations_user_{user_id}_{model_name}.csv', index=False, encoding='utf-8')
    print(f"\n✓ Рекомендации сохранены в файл: data/recommendations_user_{user_id}_{model_name}.csv")
    
    return rec_df


# ====================== ЗАПУСК ======================

if __name__ == "__main__":
    print("="*90)
    print("АНАЛИЗ И РЕКОМЕНДАЦИИ ДЛЯ ВЫПУСКНОЙ КВАЛИФИКАЦИОННОЙ РАБОТЫ")
    print("="*90)
    
    # Пункт A — Таблица и график
    plot_model_comparison()
    
    # Пункт C — Рекомендации
    print("\n" + "="*90)
    print("ПРИМЕРЫ ПЕРСОНАЛЬНЫХ РЕКОМЕНДАЦИЙ")
    print("="*90)
    
    # Рекомендации для реальных пользователей (берём первых активных)
    recommend_for_user(user_id=1, model_name='ALS', n_items=8)
    print("\n" + "-"*90)
    recommend_for_user(user_id=5, model_name='ALS', n_items=8)
    
    # Дополнительно можно раскомментировать:
    # recommend_for_user(user_id=42, model_name='ItemKNN', n_items=10)