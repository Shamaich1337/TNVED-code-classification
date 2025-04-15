# TNVED Code Classifier from Electronic Component Datasheets

Автоматическая классификация электронных компонентов по кодам ТНВЭД на основе суммаризированных datasheet'ов и модели DistilBERT.

## Цель проекта

Сократить рутинную нагрузку на сотрудников логистики компании, автоматизируя определение ТНВЭД-кодов для импортируемых электронных компонентов. Модель принимает входные данные — `part number`, `manufacturer`, описание из datasheet — и предсказывает соответствующий ТНВЭД-код.

## Используемый подход

Модель DistilBERT дообучена на корпусе с краткими описаниями электронных компонентов (summary datasheets) и метками ТН ВЭД кодов. Обработка и подготовка данных выполнена в Jupyter Notebook. Проект может быть использован как основа для API-сервиса.

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
├── notebooks/                  
│   ├── data_extraction.ipynb	# Обработка датасета и суммаризаций
│   └── model_train.ipynb		# Обучение модели
│   └── demo.py					# Демо скрипт
├── README.md
├── requirements.txt			# Зависимости
```

## Стек

- `transformers` — работа с DistilBERT

- `pandas`, `numpy`, `scikit-learn` — обработка данных и метрик

- `PyTorch` — обучение модели

  

## Метрики



|              | precision | recall | f1-score | support |
| ------------ | --------- | ------ | -------- | ------- |
| accuracy     |           |        | 0.91     | 1488    |
| macro avg    | 0.72      | 0.75   | 0.72     | 1488    |
| weighted avg | 0.93      | 0.91   | 0.92     | 1488    |

## Демо

Скрипт [`demo.py`](./demo.py) позволяет протестировать модель на произвольном текстовом описании электронного компонента и получить предсказанный ТНВЭД-код.

**Пример входного текста:**

> The ADMV1014 is a silicon germanium (SiGe) wideband microwave downconverter...

**Выход:**

> Предсказанный TNVED-код: 8542399010
