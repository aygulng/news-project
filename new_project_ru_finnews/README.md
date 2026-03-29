# Проект: NER + тональность инвестиционных новостей (RU)

Структурированная актуальная версия проекта.

## Структура

```text
new_project_ru_finnews/
  config/                         # словари/паттерны (EntityRuler)
  data/
    raw/                          # основной корпус
    labeled/                      # train-разметка (NER, sentiment)
    test/                         # независимый holdout
    processed/                    # результаты инференса
    experiments/                  # метрики и артефакты обучения
  src/
    data/                         # сборка датасетов и holdout
    ner/                          # NER (spaCy+Ruler, HF ruBERT)
    sentiment/                    # sentiment (TF-IDF, сравнение с ruBERT)
    ui/                           # Streamlit UI
  scripts/                        # вспомогательные .ps1
  docs/                           # отчёты и текст работы
```

## Основные файлы

- UI: `src/ui/app.py`
- Сборка корпуса и автоматической разметки: `src/data/build_datasets.py`
- Сборка независимого теста: `src/data/build_independent_test_set.py`
- Gold holdout разметка: `src/data/manual_gold_holdout_annotation.py`
- NER (spaCy + EntityRuler): `src/ner/run_spacy_ner_with_ruler.py`
- NER (HF ruBERT fine-tune): `src/ner/train_hf_rubert_ner.py`
- Sentiment (TF-IDF + LogisticRegression): `src/sentiment/run_entity_sentiment_tfidf.py`
- Сравнение sentiment (TF-IDF vs ruBERT): `src/sentiment/compare_entity_sentiment_tfidf_rubert.py`
- PS-скрипт запуска ruBERT обучения: `scripts/run_train_rubert_base.ps1`
- Отчёты: `docs/PROJECT_REPORT.md`, `docs/SCIENTIFIC_ARTICLE_DRAFT.md`

## Установка

```bash
python -m pip install -r requirements.txt
python -m pip install spacy
python -m spacy download ru_core_news_sm
```

`torch` устанавливается отдельно под вашу систему (см. `requirements.txt`).

## Запуск (из папки `new_project_ru_finnews`)

### 1) Собрать основной корпус + train-разметку

```bash
python src/data/build_datasets.py --main-news 1500 --ner-labels 1500 --sent-labels 1500
```

### 2) Собрать независимый holdout

```bash
python src/data/build_independent_test_set.py --test-news 300 --test-ner-labels 600 --test-sent-labels 600
```

### 3) NER: spaCy + EntityRuler

```bash
python src/ner/run_spacy_ner_with_ruler.py
```

### 4) NER: дообучение HF ruBERT-base

```bash
python src/ner/train_hf_rubert_ner.py --hf-model DeepPavlov/rubert-base-cased --epochs 8 --batch-size 4 --balance-train
```

или через PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_train_rubert_base.ps1
```

### 4b) NER: дообучение spaCy (transformer+ner)

Установите дополнение:

```bash
python -m pip install spacy-transformers
```

Пошагово:

```bash
python src/ner/prepare_spacy_ner_data.py --train-csv data/labeled/ner_entity_dataset.csv --holdout-csv data/test/test_ner_dataset.csv --output-dir data/spacy --dev-ratio 0.2 --seed 42
python src/ner/train_spacy_transformer_ner.py --train-spacy data/spacy/train.spacy --dev-spacy data/spacy/dev.spacy --output-dir data/experiments/spacy_transformer_ner --config-path data/experiments/spacy_transformer_ner/config.cfg --hf-transformer DeepPavlov/rubert-base-cased --gpu-id -1 --max-steps 12000 --eval-frequency 250 --dropout 0.15 --patience 1200 --seed 42 --run-debug-data
python src/ner/evaluate_spacy_finetuned_ner.py --model-dir data/experiments/spacy_transformer_ner/model-best --eval-csv data/test/test_ner_dataset.csv --output-dir data/experiments/spacy_transformer_ner
```

Или одним PowerShell-скриптом:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_train_spacy_transformer_ner.ps1
```

### 5) Sentiment по сущностям (TF-IDF + LR)

```bash
python src/sentiment/run_entity_sentiment_tfidf.py --dedupe-entities
```

### 6) Сравнение sentiment-моделей

```bash
python src/sentiment/compare_entity_sentiment_tfidf_rubert.py --epochs 2 --batch-size 4
```

### 7) UI

```bash
streamlit run src/ui/app.py
```

В UI можно выбрать режим:
- `Только сущности (основной режим)` — выводит только `title, entity, type`.
- `Сущности + тональность (дополнительно)` — выводит `title, entity, type, sentiment`.

## Где смотреть результаты

- Готовые предсказания: `data/processed/`
- Holdout/эксперименты и метрики: `data/experiments/`
- Размеченные train/test наборы: `data/labeled/`, `data/test/`
