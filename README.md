# Recommender System для интернет-магазина и Avito

Рекомендательная система, разработанная в рамках выпускной квалификационной работы. Включает реализацию и сравнение нескольких алгоритмов: **ALS**, **BPR**, **ItemKNN**, **BM25** и **TF-IDF**.

## 🎯 Цель работы

Разработка и сравнительный анализ рекомендательных систем для e-commerce платформ на основе **implicit feedback** (просмотры, добавления в корзину, покупки).

## 📊 Используемые датасеты

### 1. RetailRocket (интернет-магазин)

**Источник:** [RetailRocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

Данные содержат реальные взаимодействия пользователей с товарами:

- `events.csv` — события (просмотры, клики, добавления в корзину, транзакции)
- `item_properties.csv` — свойства товаров (категории, бренды и др.)
- `category_tree.csv` — иерархия категорий

### 2. Avito (платформа объявлений)

**Источник:** [Avito TechML Cup 2025](https://www.kaggle.com/datasets/halssara/avito-techml-cup-2025)

Данные для прогнозирования спроса на объявления:

- `clickstream.pq` — потоки кликов пользователей
- `cat_features.pq` — категориальные признаки
- `text_features.pq` — текстовые описания объявлений

> **Важно:** Сырые данные не включены в репозиторий. Скачайте их по ссылкам выше и разместите согласно инструкции.

## 🚀 Установка и запуск

### 1. Клонирование репозитория

```bash
git clone https://github.com/SheriShK/Recommender-system.git
cd Recommender-system/app
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

Основные библиотеки:

- `numpy`, `pandas` - обработка данных
- `scikit-learn` - TF-IDF, метрики
- `implicit` - ALS, BPR
- `lightfm` - альтернативная реализация
- `matplotlib`, `seaborn` - визуализация

### 3. Подготовка данных RetailRocket

```bash
cd for_retail
python prepare_retailrocket.py
```

Скрипт создаст в папке `data/`:

- `interactions.csv` - матрица взаимодействий
- `items.csv` - метаданные товаров
- `user_to_idx.json`, `item_to_idx.json` - маппинги индексов
- `confidence_matrix.npz` - матрица уверенности

### 4. Обучение моделей

```bash
# ALS, BPR, ItemKNN
python train_models.py

# BM25 и оптимизированный ItemKNN
python train_bm25_itemknn.py
```

### 5. Оценка и получение рекомендаций

```bash
python evaluate_and_recommend.py
```

## 📈 Результаты экспериментов

В папке `for_retail/data/` сохраняются:

- `model_comparison_results.csv` - сравнение всех моделей (Hit Rate, NDCG)
- `best_hyperparameters.json` - оптимальные гиперпараметры
- `grid_search_results.csv` - полные результаты подбора параметров
- `model_comparison.png` - визуализация сравнения моделей
- `baseline_comparison.png` - сравнение с бейзлайнами
- `hyperparameter_tuning_report.txt` - отчёт о настройке параметров

Пример результатов:


| Модель  | Hit Rate@10 | NDCG@10 | Время обучения |
| ------- | ----------- | ------- | -------------- |
| ALS     | 0.423       | 0.267   | 45 сек         |
| BPR     | 0.389       | 0.241   | 120 сек        |
| ItemKNN | 0.401       | 0.253   | 15 сек         |
| BM25    | 0.356       | 0.218   | 8 сек          |


## 🏗️ Структура проекта

```text
app/
├── for_retail/                      # RetailRocket
│   ├── prepare_retailrocket.py      # Подготовка данных
│   ├── train_models.py              # ALS, BPR, ItemKNN
│   ├── train_bm25_itemknn.py        # BM25 + оптимизация
│   ├── evaluate_and_recommend.py    # Оценка и рекомендации
│   └── data/                        # Обработанные данные (создаётся)
│       ├── interactions.csv
│       ├── best_*.pkl               # Оптимальные модели
│       ├── model_comparison_*.csv
│       └── *.png                    # Графики результатов
├── for_avito/                       # Avito
│   ├── avito_scripts/               # Скрипты для обработки
│   └── avito_data/                  # Сырые данные (скачать отдельно)
└── requirements.txt
```

## 🔄 Воспроизведение результатов

Для полного воспроизведения экспериментов:

1. Скачайте датасет RetailRocket с Kaggle.
2. Разместите файлы в папке `for_retail/archive/`.
3. Запустите подготовку: `python prepare_retailrocket.py`.
4. Обучите модели: `python train_models.py`.
5. Проведите сравнение: `python evaluate_and_recommend.py`.

Все эксперименты используют фиксированный `random_state=42` для воспроизводимости.

## 📝 Примечания

- Модели в форматах `.pkl` и `.npz` не хранятся в репозитории.
- При необходимости модели можно переобучить заново.
- Рекомендуется использовать Python 3.8+.
- Для работы с Avito потребуется дополнительная настройка (см. `avito_scripts/`).

## 👨‍🎓 Автор

ВКР по направлению "Рекомендательные системы".