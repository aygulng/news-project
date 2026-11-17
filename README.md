# Проект: NER + тональность инвестиционных новостей (RU)

Проект для магистерской работы: извлечение сущностей из русскоязычных инвестиционных новостей и анализ тональности по сущностям.

## Что входит в репозиторий

- Готовые датасеты для обучения и независимой проверки.
- Скрипты для дообучения двух NER-подходов.
- `spaCy transformer+ner` (на базе `DeepPavlov/rubert-base-cased`).
- `HF ruBERT-base` (`transformers`).
- Скрипты для sentiment (`TF-IDF + LogisticRegression`, плюс сравнение с ruBERT).
- UI на Streamlit на русском языке.

Веса моделей и чекпоинты не хранятся в git и генерируются при локальном запуске обучения.

## Структура

```text
new_project_ru_finnews/
  config/
    rare_company_index_patterns.json      # словарь/паттерны для EntityRuler
  data/
    raw/                                  # основной корпус новостей
    labeled/                              # train-разметка (NER, sentiment)
    test/                                 # независимый holdout (NER, sentiment)
    processed/                            # результаты инференса
    experiments/                          # метрики и артефакты обучения (создаются локально)
    spacy/                                # train/dev/holdout в формате .spacy (создаются локально)
  src/
    data/                                 # сборка/пересборка датасетов
    ner/                                  # обучение и инференс NER
    sentiment/                            # тональность по сущностям
    ui/                                   # Streamlit UI
  scripts/                                # PowerShell-скрипты запуска
  docs/                                   # отчеты/материалы
```

## Установка (чистое устройство)

Выполняйте из корня репозитория:

```bash
python -m pip install -r requirements.txt
python -m pip install spacy spacy-transformers
python -m spacy download ru_core_news_sm
```

`torch` ставьте под вашу ОС/CPU/GPU по инструкции PyTorch (если нужен GPU-режим).

## Быстрый старт: полный цикл

Перейдите в папку проекта:

```bash
cd new_project_ru_finnews
```

### 1) Подготовка/пересборка датасетов

Основной корпус + train-разметка:

```bash
python src/data/build_datasets.py --main-news 1500 --ner-labels 1500 --sent-labels 1500
```

Независимый holdout:

```bash
python src/data/build_independent_test_set.py --test-news 300 --test-ner-labels 600 --test-sent-labels 600
```

Ручная gold-проверка holdout (опционально):

```bash
python src/data/manual_gold_holdout_annotation.py
```

### 2) NER baseline: spaCy pretrained + EntityRuler

```bash
python src/ner/run_spacy_ner_with_ruler.py
```

### 3) NER дообучение: spaCy transformer+ner (CPU)

Пошагово:

```bash
python src/ner/prepare_spacy_ner_data.py --train-csv data/labeled/ner_entity_dataset.csv --holdout-csv data/test/test_ner_dataset.csv --output-dir data/spacy --dev-ratio 0.2 --seed 42
python src/ner/train_spacy_transformer_ner.py --train-spacy data/spacy/train.spacy --dev-spacy data/spacy/dev.spacy --output-dir data/experiments/spacy_transformer_ner --config-path data/experiments/spacy_transformer_ner/config.cfg --hf-transformer DeepPavlov/rubert-base-cased --gpu-id -1 --max-steps 12000 --eval-frequency 250 --dropout 0.15 --patience 1200 --seed 42 --run-debug-data
python src/ner/evaluate_spacy_finetuned_ner.py --model-dir data/experiments/spacy_transformer_ner/model-best --eval-csv data/test/test_ner_dataset.csv --output-dir data/experiments/spacy_transformer_ner
```

Одним PowerShell-скриптом:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_train_spacy_transformer_ner.ps1
```

### 4) NER дообучение: HF ruBERT-base

```bash
python src/ner/train_hf_rubert_ner.py --hf-model DeepPavlov/rubert-base-cased --epochs 8 --batch-size 4 --balance-train --disable-cuda
```

Или через PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_train_rubert_base.ps1
```

### 5) Тональность по сущностям (TF-IDF + LR)

```bash
python src/sentiment/run_entity_sentiment_tfidf.py --dedupe-entities
```

### 6) Сравнение sentiment-моделей (опционально)

```bash
python src/sentiment/compare_entity_sentiment_tfidf_rubert.py --epochs 2 --batch-size 4 --disable-cuda
```

### 7) UI

```bash
streamlit run src/ui/app.py
```

В UI:
- `Только сущности` - основной режим.
- `Сущности + тональность` - дополнительный режим.
- Можно выбрать NER-модель.

## Что является входом и выходом

Вход:
- `data/raw/russian_financial_news_clean_1500.csv` - основной корпус.
- `data/labeled/ner_entity_dataset.csv` - train разметка NER.
- `data/labeled/entity_sentiment_dataset.csv` - train разметка sentiment.
- `data/test/test_ner_dataset.csv` - независимый NER holdout.
- `data/test/test_entity_sentiment_dataset.csv` - независимый sentiment holdout.

Выход после запуска:
- `data/experiments/...` - модели, метрики, предсказания.
- `data/processed/...` - итоговые таблицы инференса.

## Ключевые скрипты

- `src/data/build_datasets.py` - сборка основного корпуса и train-разметки.
- `src/data/build_independent_test_set.py` - сборка независимого test/holdout.
- `src/ner/prepare_spacy_ner_data.py` - подготовка `.spacy` для spaCy training.
- `src/ner/train_spacy_transformer_ner.py` - дообучение spaCy transformer+ner.
- `src/ner/evaluate_spacy_finetuned_ner.py` - оценка spaCy на holdout.
- `src/ner/train_hf_rubert_ner.py` - дообучение HF ruBERT NER.
- `src/sentiment/run_entity_sentiment_tfidf.py` - TF-IDF+LR по тональности.
- `src/ui/app.py` - интерфейс пользователя.

## Важные заметки для нового контекста

- Это исследовательский прототип, в первую очередь для воспроизводимого эксперимента.
- Датасеты в git включены, тяжёлые веса моделей в git не включены.
- При первом запуске обучение может быть долгим на CPU.
- Для Windows PowerShell многострочные команды пишите через `` ` `` (обратная кавычка), не через `^`.
- Подробная методология и результаты:
- `docs/PROJECT_REPORT.md`
- `docs/SCIENTIFIC_ARTICLE_DRAFT.md`
- `docs/PRESENTATION_RUBERT_NER.md`
