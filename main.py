import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from scipy.sparse import csr_matrix, save_npz
import json
import pickle
import os

# Инициализация
fake = Faker('ru_RU')
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# ========== ПАРАМЕТРЫ ==========
NUM_USERS = 2000
NUM_ITEMS = 500
NUM_INTERACTIONS = 30000
MIN_INTERACTIONS_PER_USER = 2
MAX_INTERACTIONS_PER_USER = 30

# Категории объявлений
CATEGORIES = {
    1: "Электроника",
    2: "Недвижимость",
    3: "Транспорт",
    4: "Услуги",
    5: "Личные вещи",
    6: "Для дома и дачи",
    7: "Работа",
    8: "Хобби и отдых"
}

# Города
CITIES = ['Москва', 'Санкт-Петербург', 'Новосибирск', 'Екатеринбург', 'Казань',
          'Нижний Новгород', 'Красноярск', 'Челябинск', 'Самара', 'Уфа']

# Типы действий и их веса
ACTION_TYPES = {
    'view': 1,
    'like': 3,
    'contact': 4,
    'purchase': 5
}

def generate_users(n):
    users = []
    for i in range(n):
        user_id = i + 1
        preferred_categories = random.sample(list(CATEGORIES.keys()), k=random.randint(1, 3))
        user = {
            'user_id': user_id,
            'name': fake.name(),
            'age': random.randint(18, 70),
            'city': random.choice(CITIES),
            'registration_date': fake.date_between(start_date='-2y', end_date='today'),
            'preferred_categories': preferred_categories,
            'user_emb': np.random.normal(0, 1, 20).tolist()
        }
        users.append(user)
    return pd.DataFrame(users)

def generate_items(n):
    items = []
    for i in range(n):
        item_id = i + 1
        category_id = random.choice(list(CATEGORIES.keys()))
        if category_id == 2:
            price = random.randint(10000, 50000000)
        elif category_id == 3:
            price = random.randint(50000, 3000000)
        else:
            price = random.randint(500, 150000)
        item = {
            'item_id': item_id,
            'title': fake.sentence(nb_words=3),
            'category_id': category_id,
            'category_name': CATEGORIES[category_id],
            'price': price,
            'city': random.choice(CITIES),
            'created_date': fake.date_between(start_date='-1y', end_date='today'),
            'views_count': random.randint(0, 1000),
            'item_emb': np.random.normal(0, 1, 20).tolist()
        }
        items.append(item)
    return pd.DataFrame(items)

def generate_interactions(users_df, items_df, num_interactions):
    interactions = []
    users_list = users_df.to_dict('records')
    items_list = items_df.to_dict('records')
    
    for user in users_list:
        if isinstance(user['preferred_categories'], str):
            user['preferred_categories'] = eval(user['preferred_categories'])
    
    for _ in range(num_interactions):
        user = random.choice(users_list)
        user_id = user['user_id']
        user_age = user['age']
        user_city = user['city']
        preferred_cats = user['preferred_categories']
        
        weights = []
        for item in items_list:
            weight = 1.0
            if item['city'] == user_city:
                weight *= 2.0
            if item['category_id'] in preferred_cats:
                weight *= 3.0
            if user_age < 25 and item['price'] > 50000:
                weight *= 0.3
            days_old = (datetime.now().date() - item['created_date']).days
            if days_old < 7:
                weight *= 1.5
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        item = np.random.choice(items_list, p=weights)
        item_id = item['item_id']
        
        action_type = np.random.choice(
            list(ACTION_TYPES.keys()),
            p=[0.6, 0.2, 0.15, 0.05]
        )
        relevance = ACTION_TYPES[action_type]
        interaction_date = fake.date_between(start_date='-90d', end_date='today')
        
        interaction = {
            'user_id': user_id,
            'item_id': item_id,
            'action_type': action_type,
            'relevance': relevance,
            'timestamp': interaction_date
        }
        interactions.append(interaction)
    
    return pd.DataFrame(interactions)

def generate_user_features(users_df):
    city_dummies = pd.get_dummies(users_df['city'], prefix='city')
    users_df['age_group'] = pd.cut(users_df['age'],
                                    bins=[0, 25, 35, 50, 100],
                                    labels=['18-25', '26-35', '36-50', '50+'])
    age_dummies = pd.get_dummies(users_df['age_group'], prefix='age')
    
    cat_features = []
    for cats in users_df['preferred_categories']:
        row = [0] * len(CATEGORIES)
        for c in cats:
            row[c-1] = 1
        cat_features.append(row)
    cat_df = pd.DataFrame(cat_features, columns=[f'pref_cat_{i}' for i in range(1, len(CATEGORIES)+1)])
    
    user_features = pd.concat([users_df[['user_id']], city_dummies, age_dummies, cat_df], axis=1)
    return user_features

def generate_item_features(items_df):
    cat_dummies = pd.get_dummies(items_df['category_id'], prefix='cat')
    city_dummies = pd.get_dummies(items_df['city'], prefix='city')
    
    items_df['price_group'] = pd.cut(items_df['price'],
                                      bins=[0, 1000, 10000, 50000, 100000, 1e9],
                                      labels=['budget', 'medium', 'premium', 'luxury', 'ultra'])
    price_dummies = pd.get_dummies(items_df['price_group'], prefix='price')
    
    item_features = pd.concat([items_df[['item_id']], cat_dummies, city_dummies, price_dummies], axis=1)
    return item_features

def create_interaction_matrix(interactions_df, users_df, items_df):
    n_users = len(users_df)
    n_items = len(items_df)
    
    # Ключевое исправление: преобразуем в обычный int
    user_to_idx = {int(uid): i for i, uid in enumerate(users_df['user_id'].values)}
    item_to_idx = {int(iid): j for j, iid in enumerate(items_df['item_id'].values)}
    
    rows = []
    cols = []
    data_binary = []
    data_confidence = []
    
    for _, row in interactions_df.iterrows():
        u_idx = user_to_idx[int(row['user_id'])]
        i_idx = item_to_idx[int(row['item_id'])]
        
        rows.append(u_idx)
        cols.append(i_idx)
        data_binary.append(1)
        data_confidence.append(row['relevance'])
    
    interaction_matrix = csr_matrix((data_binary, (rows, cols)),
                                     shape=(n_users, n_items))
    confidence_matrix = csr_matrix((data_confidence, (rows, cols)),
                                    shape=(n_users, n_items))
    
    return interaction_matrix, confidence_matrix, user_to_idx, item_to_idx

def save_all_data():
    print("Генерация данных для рекомендательной системы...")
    
    os.makedirs('data', exist_ok=True)
    
    users_df = generate_users(NUM_USERS)
    print(f"✓ Сгенерировано {NUM_USERS} пользователей")
    
    items_df = generate_items(NUM_ITEMS)
    print(f"✓ Сгенерировано {NUM_ITEMS} объявлений")
    
    interactions_df = generate_interactions(users_df, items_df, NUM_INTERACTIONS)
    print(f"✓ Сгенерировано {len(interactions_df)} взаимодействий")
    
    user_features = generate_user_features(users_df)
    item_features = generate_item_features(items_df)
    
    interaction_matrix, confidence_matrix, user_to_idx, item_to_idx = create_interaction_matrix(
        interactions_df, users_df, items_df
    )
    
    users_df.to_csv('data/users.csv', index=False)
    items_df.to_csv('data/items.csv', index=False)
    interactions_df.to_csv('data/interactions.csv', index=False)
    user_features.to_csv('data/user_features.csv', index=False)
    item_features.to_csv('data/item_features.csv', index=False)
    
    save_npz('data/interaction_matrix.npz', interaction_matrix)
    save_npz('data/confidence_matrix.npz', confidence_matrix)
    
    # Теперь JSON работает, так как ключи — обычные int
    with open('data/user_to_idx.json', 'w') as f:
        json.dump(user_to_idx, f)
    with open('data/item_to_idx.json', 'w') as f:
        json.dump(item_to_idx, f)
    
    # Бэкап в pickle
    with open('data/user_to_idx.pkl', 'wb') as f:
        pickle.dump(user_to_idx, f)
    with open('data/item_to_idx.pkl', 'wb') as f:
        pickle.dump(item_to_idx, f)
    
    print("\n" + "="*50)
    print("СТАТИСТИКА ДАННЫХ:")
    print(f"Плотность матрицы: {interaction_matrix.nnz / (NUM_USERS * NUM_ITEMS) * 100:.2f}%")
    print(f"Среднее взаимодействий на пользователя: {interaction_matrix.nnz / NUM_USERS:.1f}")
    print(f"Среднее взаимодействий на объявление: {interaction_matrix.nnz / NUM_ITEMS:.1f}")
    
    action_dist = interactions_df['action_type'].value_counts()
    print("\nРаспределение действий:")
    for action, count in action_dist.items():
        print(f"  {action}: {count} ({count/len(interactions_df)*100:.1f}%)")
    
    print("\n✓ Все данные сохранены в папку 'data/'")
    
    return {
        'users': users_df,
        'items': items_df,
        'interactions': interactions_df,
        'interaction_matrix': interaction_matrix,
        'confidence_matrix': confidence_matrix,
        'user_features': user_features,
        'item_features': item_features
    }

if __name__ == "__main__":
    data = save_all_data()