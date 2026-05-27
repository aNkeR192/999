```markdown
# Airsoft Equipment Classification API

API для определения категории и подкатегории страйкбольного снаряжения по тексту объявления и фотографиям.

## Данные

### posts.parquet
- 66 370 постов
- Колонки: `Id`, `CategoryId`, `categoryname`, `Text`
- 3 категории: Снаряжение и защита, Аксессуары и Запчасти, Страйкбольное оружие

### photos.parquet
- 96 500 фото
- Колонки: `Id`, `Url`, `DataSource`, `PostId`

### subcategory_images.zip
- Папки: ak, HK, M serias, helmet, vest, pouch, backpack, machinegun, pistol, rifle, shotgun

## Архитектура

```
Текст → DistilBERT → Категория (3 класса)
Фото → ResNet50 → Подкатегория (11 классов)
Объединение → JSON с предсказаниями
```

## Запуск

```bash
pip install -r requirements.txt
python train_subcategory.py --images data/processed/images/dataset
python train_category.py --posts posts.parquet
python app.py
```

## API

`POST /predict`

Заголовок: `X-API-Key: airsoft-demo-key-2024`

Тело запроса:
```json
{
  "post_id": "123",
  "text": "Продаю автомат AK",
  "photos": [{"photo_id": "1", "url": "https://example.com/photo.jpg"}]
}
```

Ответ:
```json
{
  "post_id": "123",
  "predictions": [
    {
      "object_id": "1",
      "category": "Страйкбольное оружие",
      "subcategory": "AK",
      "confidence": 0.95,
      "photo_ids": ["1"]
    }
  ]
}
```

## Интерфейс

- Swagger: `http://localhost:8000/docs`
- Gradio: `http://localhost:7860`

## Деплой на Hugging Face Spaces

1. Создайте Space с SDK Gradio
2. Загрузите файлы проекта
3. Добавьте секрет `API_KEY`
```
