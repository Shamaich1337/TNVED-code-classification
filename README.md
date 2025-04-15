# TNVED Code Classifier from Electronic Component Datasheets

Автоматическая классификация электронных компонентов по кодам ТНВЭД на основе суммаризированных datasheet'ов и модели DistilBERT.

## Цель проекта

Сократить рутинную нагрузку на сотрудников логистики компании, автоматизируя определение ТНВЭД-кодов для импортируемых электронных компонентов. Модель принимает входные данные — `part number`, `manufacturer`, описание из datasheet — и предсказывает соответствующий ТНВЭД-код.

## Используемый подход

Модель DistilBERT дообучена на корпусе с краткими описаниями электронных компонентов (summary datasheets) и метками ТНВЭД. Обработка и подготовка данных выполнена в Jupyter Notebook. Проект может быть использован как основа для API-сервиса.

## Структура проекта

```
.
├── best_model_distilbert/      # Финальная обученная модель (HuggingFace format)
│   ├── config.json
│   └── model.safetensors
├── data/                       # Данные для обучения и inference
│   ├── datasheets/             # Суммаризированные datasheets (txt)
│   ├── data.csv                # Очищенный csv
│   └── df_clear_uniq_3.csv		# Исходный csv
│   └── label_encoder.joblib	# Энкодер для преобразовния ответов 
├── notebooks/                  # Jupyter ноутбуки с пайплайном
│   ├── data_extraction.ipynb
│   └── model_train.ipynb
├── README.md
├── requirements.txt
```

## Стек

- `transformers`, `datasets`, `tokenizers` — работа с DistilBERT

- `pandas`, `numpy`, `scikit-learn` — обработка данных и метрик

- `PyTorch` — обучение модели

  

## Как использовать

1. Установить зависимости:

```bash
pip install -r requirements.txt
```

2. Запустить ноутбук из папки `notebooks/`, чтобы:

- провести inference
- дообучить модель на новых данных
- оценить качество модели

3. Модель `best_model_distilbert/` можно загрузить через `transformers`:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("best_model_distilbert")
tokenizer = AutoTokenizer.from_pretrained("best_model_distilbert")
```

## 🛠️ Запуск inference

Пример кода:

```python
def predict_tnved(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(dim=-1).item()
    return predicted_class
```

