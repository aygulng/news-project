# Отчёт по проекту: NER и тональность инвестиционных новостей (RU)

## 1. Цель проекта

Собрать и запустить рабочий прототип для магистерской темы:  
**«Применение NLP-моделей для извлечения сущностей и анализа тональности инвестиционных новостей на русском языке»**.

Практическая цель этапа:
- получить чистый корпус русскоязычных финансовых новостей;
- подготовить размеченные наборы для NER и тональности по сущностям;
- реализовать baseline-решение для извлечения сущностей;
- дообучить HF-модель NER и сравнить её с baseline;
- посчитать метрики на независимом holdout-наборе.

---

## 2. Что было сделано

### 2.1 Создан отдельный проект

Проект вынесен в отдельную директорию:
- `new_project_ru_finnews/`

Это позволило не смешивать эксперименты с предыдущими файлами.

### 2.2 Сформирован основной корпус новостей

Источник:
- `Kasymkhan/RussianFinancialNews` (HuggingFace Datasets).

Скрипт:
- `new_project_ru_finnews/src/data/build_datasets.py`

Что делает скрипт:
- загружает исходный датасет;
- очищает текст (HTML, URL, лишние пробелы);
- удаляет записи с пустым заголовком/текстом и `no title`;
- убирает дубли по точным и шаблонным сигнатурам;
- формирует итоговый корпус.

Результат:
- `new_project_ru_finnews/data/raw/russian_financial_news_clean_1500.csv`
- `new_project_ru_finnews/data/raw/russian_financial_news_clean_1500.jsonl`

Итог корпуса:
- `1500` новостей;
- пустые заголовки и пустые тексты удалены.

### 2.3 Подготовлены размеченные наборы

Базовые скрипты:
- `new_project_ru_finnews/src/data/build_datasets.py`
- `new_project_ru_finnews/src/data/rebuild_ner_datasets.py`

Сформированы:
- `new_project_ru_finnews/data/labeled/ner_entity_dataset.csv`
- `new_project_ru_finnews/data/labeled/entity_sentiment_dataset.csv`

Актуальные типы сущностей в NER-наборе:
- `Company`
- `Index`
- `Event`
- `Commodity`
- `Currency`
- `Person`
- `Location`
- `Sector`
- `Time`

Актуальный размер NER train-разметки:
- `11278` строк (`data/processed/ner_rebuild_summary.json`).

### 2.4 Собран независимый test (holdout) без пересечений

Скрипт формирования независимого holdout-корпуса:
- `new_project_ru_finnews/src/data/build_independent_test_set.py`

Результаты:
- `new_project_ru_finnews/data/test/test_corpus_clean_300.csv`
- `new_project_ru_finnews/data/test/test_corpus_clean_300.jsonl`
- `new_project_ru_finnews/data/test/test_overlap_report.json`

Проверка пересечений:
- `overlap_exact_count = 0`
- `overlap_pattern_count = 0`

После пересборки NER-разметки на этом holdout-корпусе (`rebuild_ner_datasets.py`) получен актуальный тестовый NER-набор:
- `new_project_ru_finnews/data/test/test_ner_dataset.csv` (`858` строк).

### 2.5 Реализован baseline NER: spaCy + EntityRuler

Скрипт:
- `new_project_ru_finnews/src/ner/run_spacy_ner_with_ruler.py`

Использовано:
- `spaCy` модель `ru_core_news_sm` для базовых сущностей;
- `EntityRuler` для финансовых и редких форм.

Паттерны `EntityRuler`:
- `new_project_ru_finnews/config/rare_company_index_patterns.json`

Выходные файлы:
- `new_project_ru_finnews/data/processed/spacy_ruler_ner_predictions.csv`
- `new_project_ru_finnews/data/processed/spacy_ruler_eval_predictions.csv`
- `new_project_ru_finnews/data/processed/spacy_ruler_eval_metrics.json`
- `new_project_ru_finnews/data/processed/spacy_ruler_summary.json`

### 2.6 Реализован анализ тональности по сущностям (TF-IDF + Logistic Regression)

Скрипт:
- `new_project_ru_finnews/src/sentiment/run_entity_sentiment_tfidf.py`

Что делает:
- обучает классификатор тональности на `entity_sentiment_dataset.csv`;
- строит признаки из контекста, сущности и её типа;
- предсказывает тональность для каждой извлечённой сущности;
- формирует итоговую таблицу `title, entity, type, sentiment`.

Выходные файлы:
- `new_project_ru_finnews/data/processed/news_entity_sentiment_tfidf.csv`
- `new_project_ru_finnews/data/processed/sentiment_tfidf_metrics.json`

### 2.7 Добавлено сравнение sentiment-моделей: TF-IDF vs RuBERT-base

Скрипт:
- `new_project_ru_finnews/src/sentiment/compare_entity_sentiment_tfidf_rubert.py`

Назначение:
- обучает и оценивает две модели на одном test-наборе (`test_entity_sentiment_dataset.csv`);
- сохраняет метрики и предсказания для анализа ошибок.

Выходные файлы:
- `new_project_ru_finnews/data/experiments/sentiment_tfidf_rubert/tfidf_test_metrics.json`
- `new_project_ru_finnews/data/experiments/sentiment_tfidf_rubert/rubert_test_metrics.json`
- `new_project_ru_finnews/data/experiments/sentiment_tfidf_rubert/sentiment_test_predictions_compare.csv`
- `new_project_ru_finnews/data/experiments/sentiment_tfidf_rubert/sentiment_comparison_summary.json`

---

## 3. Что использовали

### 3.1 Библиотеки

- `datasets` (HuggingFace datasets)
- `pandas`, `numpy`
- `spacy`
- `transformers`, `torch`
- `scikit-learn`
- `regex`/`re`, `json`, `pathlib`

### 3.2 Модели и подход

- NER baseline: `spaCy ru_core_news_sm + EntityRuler`
- NER model: `HF ruBERT-base (DeepPavlov/rubert-base-cased), fine-tuned`
- Entity sentiment model: `TF-IDF + Logistic Regression`
- Дополнительное сравнение sentiment: `RuBERT-base (sequence classification)`
- Оценка качества: `precision`, `recall`, `f1-score`, `accuracy`, `macro avg`, `weighted avg`

---

## 4. Как получены результаты

Базовые шаги запуска:

```bash
python -m pip install -r requirements.txt
python -m spacy download ru_core_news_sm
python new_project_ru_finnews/src/data/build_datasets.py
python new_project_ru_finnews/src/data/build_independent_test_set.py
python new_project_ru_finnews/src/data/rebuild_ner_datasets.py
python new_project_ru_finnews/src/ner/run_spacy_ner_with_ruler.py
python new_project_ru_finnews/src/ner/train_hf_rubert_ner.py --train-csv new_project_ru_finnews/data/labeled/ner_entity_dataset.csv --test-csv new_project_ru_finnews/data/test/test_ner_dataset.csv --output-dir new_project_ru_finnews/data/experiments/hf_rubert_base_ner_v2_cpu --hf-model DeepPavlov/rubert-base-cased --epochs 8 --batch-size 4 --learning-rate 2e-5 --weight-decay 0.01 --warmup-ratio 0.1 --max-length 256 --stride 96 --balance-train --disable-cuda
python new_project_ru_finnews/src/sentiment/run_entity_sentiment_tfidf.py --dedupe-entities
python new_project_ru_finnews/src/sentiment/compare_entity_sentiment_tfidf_rubert.py --epochs 2 --batch-size 4 --learning-rate 2e-5 --max-length 192
```

После запуска:
- формируется основной корпус;
- формируются/пересобираются размеченные наборы;
- формируется независимый holdout;
- запускается NER baseline и HF NER;
- считается качество на holdout;
- формируются результаты по тональности и сравнение моделей.

---

## 5. Структура проекта (папки и файлы)

```text
new_project_ru_finnews/
  config/
    rare_company_index_patterns.json
  data/
    raw/
      russian_financial_news_clean_1500.csv
      russian_financial_news_clean_1500.jsonl
    labeled/
      ner_entity_dataset.csv
      entity_sentiment_dataset.csv
    test/
      test_corpus_clean_300.csv
      test_corpus_clean_300.jsonl
      test_ner_dataset.csv
      test_entity_sentiment_dataset.csv
      test_overlap_report.json
    processed/
      ner_rebuild_summary.json
      spacy_ruler_ner_predictions.csv
      spacy_ruler_eval_predictions.csv
      spacy_ruler_eval_metrics.json
      spacy_ruler_summary.json
      sentiment_tfidf_metrics.json
      news_entity_sentiment_tfidf.csv
    experiments/
      hf_rubert_base_ner_v2_cpu/
      sentiment_tfidf_rubert/
  docs/
    PROJECT_REPORT.md
    SCIENTIFIC_ARTICLE_DRAFT.md
    IMPORTANT_LOGS_FOR_RESEARCH.md
  scripts/
    run_train_rubert_base.ps1
  src/
    data/
    ner/
    sentiment/
    ui/
```

---

## 6. Полученные результаты (текущее состояние)

### 6.1 Данные

- train-корпус: `1500` новостей;
- holdout-корпус: `300` новостей;
- train NER-разметка: `11278` строк;
- holdout NER-разметка: `858` строк;
- train sentiment-набор: `1500` строк;
- test sentiment-набор: `274` строки.

### 6.2 Качество NER на независимом holdout

По файлу `spacy_ruler_eval_metrics.json`:
- `accuracy = 0.7179`
- `macro F1 = 0.5872`
- `weighted F1 = 0.7287`

F1 по классам:
- `Company = 0.9969`
- `Index = 1.0000`
- `Event = 0.9163`
- `Location = 0.9908`
- `Commodity = 1.0000`
- `Sector = 0.9684`
- `Currency = 0.0000`
- `Person = 0.0000`
- `Time = 0.0000`

### 6.3 Объём предсказанных сущностей в основном корпусе

По `spacy_ruler_summary.json`:
- обработано документов: `1500`
- извлечено сущностей: `4417`
- источник сущности:
  - `spacy: 2329`
  - `ruler: 2088`

### 6.4 Результат дообучения HF ruBERT-base (token classification)

Скрипт:
- `new_project_ru_finnews/src/ner/train_hf_rubert_ner.py`

Актуальная конфигурация запуска:
- модель: `DeepPavlov/rubert-base-cased`
- эпохи: `8`
- batch size: `4`
- learning rate: `2e-5`
- max length: `256`
- балансировка train-документов: включена (`--balance-train`)
- устройство: `CPU` (`--disable-cuda`)

Актуальные артефакты:
- `new_project_ru_finnews/data/experiments/hf_rubert_base_ner_v2_cpu/hf_model/`
- `new_project_ru_finnews/data/experiments/hf_rubert_base_ner_v2_cpu/hf_train_metrics.json`
- `new_project_ru_finnews/data/experiments/hf_rubert_base_ner_v2_cpu/hf_holdout_metrics.json`
- `new_project_ru_finnews/data/experiments/hf_rubert_base_ner_v2_cpu/hf_holdout_predictions.csv`
- `new_project_ru_finnews/data/experiments/hf_rubert_base_ner_v2_cpu/hf_training_summary.json`

Holdout-метрики HF ruBERT-base:
- `accuracy = 0.9627`
- `macro F1 = 0.8727`
- `weighted F1 = 0.9772`

Сравнение с baseline `spaCy + EntityRuler` на том же holdout:
- baseline: `macro F1 = 0.5872`
- HF ruBERT-base: `macro F1 = 0.8727`
- разница: `+0.2854` в пользу HF ruBERT-base.

---

## 6.5 Подробное обучение двух моделей

### Модель 1: spaCy pretrained + EntityRuler (baseline)

Тип модели:
- `spaCy ru_core_news_sm` + словарные правила `EntityRuler`.
- Дообучение весов не выполнялось; это rule-enhanced baseline.

Скрипт:
- `new_project_ru_finnews/src/ner/run_spacy_ner_with_ruler.py`

Входные данные:
- корпус для извлечения: `new_project_ru_finnews/data/raw/russian_financial_news_clean_1500.csv`
- holdout для оценки: `new_project_ru_finnews/data/test/test_ner_dataset.csv`

Параметры запуска:
- `--spacy-model ru_core_news_sm`
- `--ruler-patterns new_project_ru_finnews/config/rare_company_index_patterns.json`
- `--max-docs 0` (обработка полного корпуса)

Как считается совпадение сущности на holdout:
- `3` балла: точное совпадение;
- `2` балла: одна строка содержит другую (при длине >= 4);
- `1` балл: пересечение токенов `>= 0.6`;
- используется лучший кандидат для сравнения `pred_type` с `gold_type`.

Итог качества (holdout):
- `accuracy = 0.7179`
- `macro F1 = 0.5872`
- `weighted F1 = 0.7287`

### Модель 2: HF ruBERT-base (fine-tuned, token classification)

Тип модели:
- `BertForTokenClassification` с BIO-разметкой.
- Базовый чекпоинт: `DeepPavlov/rubert-base-cased`.

Скрипт:
- `new_project_ru_finnews/src/ner/train_hf_rubert_ner.py`

Входные данные:
- train labels: `new_project_ru_finnews/data/labeled/ner_entity_dataset.csv`
- test labels: `new_project_ru_finnews/data/test/test_ner_dataset.csv`

Подготовка train перед обучением (по `hf_training_summary.json`):
- исходно разметки: `11278` строк;
- после группировки по контекстам: `4443` документа;
- valid: `889` документов;
- train до балансировки: `3554` документа;
- train после `--balance-train`: `10521` документа.

Целевые классы:
- `Company`
- `Index`
- `Event`
- `Commodity`
- `Currency`
- `Person`
- `Location`
- `Sector`
- `Time`

Ключевые гиперпараметры:
- `epochs = 8`
- `batch_size = 4`
- `learning_rate = 2e-5`
- `weight_decay = 0.01`
- `warmup_ratio = 0.1`
- `max_length = 256`
- `stride = 96`
- `load_best_model_at_end = true`

Режим выполнения:
- устройство: `CPU` (в текущем окружении без CUDA).

Результаты:
- validation в конце обучения: `eval_f1 = 0.9779`, `eval_accuracy = 0.9869`
- holdout: `accuracy = 0.9627`
- holdout: `macro F1 = 0.8727`
- holdout: `weighted F1 = 0.9772`

Артефакты обучения:
- `new_project_ru_finnews/data/experiments/hf_rubert_base_ner_v2_cpu/hf_model/`
- `new_project_ru_finnews/data/experiments/hf_rubert_base_ner_v2_cpu/hf_train_metrics.json`
- `new_project_ru_finnews/data/experiments/hf_rubert_base_ner_v2_cpu/hf_holdout_metrics.json`
- `new_project_ru_finnews/data/experiments/hf_rubert_base_ner_v2_cpu/hf_holdout_predictions.csv`
- `new_project_ru_finnews/data/experiments/hf_rubert_base_ner_v2_cpu/hf_training_summary.json`

Готовый автоматизированный запуск (legacy-путь вывода):
- `new_project_ru_finnews/scripts/run_train_rubert_base.ps1`

---

## 6.6 Результаты модели тональности по сущностям

Модель:
- `TF-IDF + Logistic Regression` (скрипт `run_entity_sentiment_tfidf.py`).

Разбиение train/valid:
- `train_rows = 1200`
- `valid_rows = 300`

Качество на valid:
- `accuracy = 0.8867`
- `macro F1 = 0.8394`
- `weighted F1 = 0.8860`

Поклассовые F1:
- `Negative = 0.8395`
- `Neutral = 0.9231`
- `Positive = 0.7556`

Итоговый output для основного корпуса:
- файл `new_project_ru_finnews/data/processed/news_entity_sentiment_tfidf.csv`
- строк: `4058`
- распределение тональностей:
  - `Neutral: 3103`
  - `Positive: 530`
  - `Negative: 425`

---

## 6.7 Сравнение моделей тональности: TF-IDF vs RuBERT-base

Датасет сравнения:
- `new_project_ru_finnews/data/test/test_entity_sentiment_dataset.csv` (`274` примера).

Модели:
- baseline: `TF-IDF + Logistic Regression`;
- transformer: `RuBERT-base (DeepPavlov/rubert-base-cased)` в постановке sequence classification.

Параметры RuBERT (фактический запуск):
- `epochs = 2`
- `batch_size = 4`
- `learning_rate = 2e-5`
- `max_length = 192`
- устройство: `CPU`

Результаты:
- TF-IDF: `accuracy = 0.8650`, `macro F1 = 0.6374`, `weighted F1 = 0.8516`
- RuBERT-base: `accuracy = 0.9015`, `macro F1 = 0.7730`, `weighted F1 = 0.8919`

Прирост RuBERT-base:
- `+0.0365` по accuracy
- `+0.1356` по macro F1
- `+0.0403` по weighted F1

Интерпретация:
- RuBERT-base заметно лучше распознаёт тональность на независимом test-наборе;
- TF-IDF остаётся полезным baseline и быстрым сравнительным ориентиром.

---

## 7. Ограничения и интерпретация

- NER-набор пока сформирован полуавтоматически; для академически строгой оценки нужен полностью ручной gold-holdout.
- Классы NER несбалансированы, что может влиять на macro-метрики и поклассовую стабильность.
- Обучение NER и части sentiment-экспериментов выполнялось на CPU, что ограничивает объём гиперпараметрического поиска.
- Рост `eval_loss` на поздних эпохах при росте `eval_f1` указывает на возможную перекалибровку уверенности модели; рекомендуется добавить `early stopping` и калибровку вероятностей.

---

## 8. Вывод

На текущем этапе собран рабочий прототип полного цикла для NER и анализа тональности по сущностям:
- создан чистый русскоязычный финансовый корпус;
- сформированы актуальные train/test наборы;
- реализован baseline NER (`spaCy + EntityRuler`);
- выполнено дообучение NER-модели `HF ruBERT-base` с существенным приростом качества на holdout;
- реализован анализ тональности по сущностям (`TF-IDF + Logistic Regression`);
- проведено сравнение sentiment-моделей (`TF-IDF` vs `RuBERT-base`);
- все ключевые артефакты и логи сохранены для воспроизводимости.

Следующий логичный шаг:
- провести полностью ручную gold-разметку holdout и повторить сравнение моделей на одном строгом протоколе.
