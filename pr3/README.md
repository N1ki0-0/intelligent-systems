# Классификация военных и гражданских самолётов

В работе сравниваются две мультимодальные модели — **Qwen2.5-VL 3B** и **Gemma 3 4B**. Каждая получает одинаковые 30 фотографий самолётов: 15 гражданских и 15 военных. Модели работают локально через Ollama API.

## Данные и методика

Список фотографий, правильные классы, сложность и ссылки на источники Wikimedia Commons находятся в `data/aircraft_manifest.csv`. Моделям передаются только фотографии и одинаковая инструкция; правильные ответы и названия самолётов в запрос не входят.

Сравниваются точность (Accuracy), Precision, Recall, macro-F1, матрицы ошибок и время ответа. Отдельно оценивается соблюдение формата JSON: иногда модель правильно указывает `label`, но не завершает поле с объяснением. В таком случае класс можно однозначно извлечь, но сам JSON остаётся некорректным. Исходные ответы сохраняются без редактирования.

## Запуск

1. Установить [Ollama](https://ollama.com/) и скачать модели:

   ```bash
   ollama pull qwen2.5vl:3b
   ollama pull gemma3:4b
   ```

2. Открыть папку проекта в VS Code, создать окружение и установить зависимости:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. Открыть `aircraft_classification.ipynb`, выбрать интерпретатор из `.venv` и выполнить **Run All**. После выполнения сохранить ноутбук.

## Файлы

- `aircraft_classification.ipynb` — эксперимент, таблицы, графики и разбор ошибок.
- `aircraft_eval.py` — загрузка фотографий, Ollama API и расчёт метрик.
- `data/aircraft_manifest.csv` — данные и источники.
- `outputs/predictions.csv` — ответы моделей и их статусы.
- `outputs/metrics.csv` — основные метрики.
- `outputs/classification_vs_format.csv` — точность извлечённых меток и корректность JSON.
- `outputs/credits.csv` — источники и сведения о фотографиях.
- `outputs/confusion_matrices.png`, `outputs/model_comparison.png` — графики.

Файл `outputs/download_errors.csv` создаётся только при ошибках скачивания.