# Recommender System для интернет-магазина и Avito

Рекомендательная система, разработанная в рамках выпускной квалификационной работы. Включает реализацию и сравнение нескольких алгоритмов: ALS, BPR, ItemKNN, BM25 и TF-IDF.

## Цель работы

Разработка и сравнительный анализ рекомендательных систем для e-commerce платформ на основе implicit feedback (просмотры, добавления в корзину, покупки).

## Используемые датасеты

### 1. RetailRocket (интернет-магазин)
**Источник:** [RetailRocket E-commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

Данные содержат реальные взаимодействия пользователей с товарами:
- `events.csv` - события (просмотры, клики, добавления в корзину, транзакции)
- `item_properties.csv` - свойства товаров (категории, бренды и др.)
- `category_tree.csv` - иерархия категорий

### 2. Avito (платформа объявлений)
**Источник:** [Avito TechML Cup 2025]([https://www.kaggle.com/competitions/avito-demand-prediction](https://www.kaggle.com/datasets/halssara/avito-techml-cup-2025))

Данные для прогнозирования спроса на объявления:
- `clickstream.pq` - потоки кликов пользователей
- `cat_features.pq` - категориальные признаки
- `text_features.pq` - текстовые описания объявлений

> **Важно:** Сырые данные не включены в репозиторий. Скачайте их по ссылкам выше и разместите согласно инструкции.

## 🚀 Установка и запуск

### 1. Клонирование репозитория

```bash
git clone https://github.com/SheriShK/Recommender-system.git
cd Recommender-system/app
