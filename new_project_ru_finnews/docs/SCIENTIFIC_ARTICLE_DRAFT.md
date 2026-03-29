# Применение NLP-моделей для извлечения сущностей и анализа тональности в русскоязычных инвестиционных новостях

## Аннотация
В работе представлен воспроизводимый прототип пайплайна для обработки русскоязычных инвестиционных новостей: извлечение сущностей (NER) и анализ тональности по сущностям. Для NER сравниваются два подхода: `spaCy ru_core_news_sm + EntityRuler` (baseline) и fine-tuned `DeepPavlov/rubert-base-cased` в задаче token classification (BIO-разметка). Для тональности реализован классификатор `TF-IDF + Logistic Regression`, а также выполнено сравнение с RuBERT (sequence classification).

Эксперименты проведены на корпусе `Kasymkhan/RussianFinancialNews` после очистки и дедупликации. NER-оценка выполнена на независимом holdout (300 новостей, 858 сущностей). Лучший результат показал fine-tuned ruBERT-base: `accuracy = 0.9627`, `macro F1 = 0.8727`, `weighted F1 = 0.9772` против baseline `accuracy = 0.7179`, `macro F1 = 0.5872`, `weighted F1 = 0.7287`. Анализ логов обучения показывает отсутствие критического переобучения, но указывает на рост `eval_loss` на поздних эпохах (признак роста уверенности модели).

**Ключевые слова:** NLP, NER, анализ тональности, инвестиционные новости, русский язык, ruBERT, spaCy, EntityRuler, TF-IDF, Logistic Regression.

---

## 1. Введение
Финансовые новости содержат оперативные сигналы для инвестиционного анализа, но представлены в неструктурированном виде. Для автоматизации анализа необходимо:
1. извлекать ключевые сущности (компании, индексы, события, валюты, сырьё, локации и др.);
2. определять тональность упоминаний сущностей в контексте.

Целью исследования является разработка и оценка прототипа NER + entity-level sentiment для русскоязычных инвестиционных новостей.

---

## 2. Данные

### 2.1 Источник
Использован датасет HuggingFace: `Kasymkhan/RussianFinancialNews`.

### 2.2 Подготовка корпуса
Предобработка:
1. удаление HTML, URL, служебных символов;
2. нормализация пробелов;
3. удаление пустых записей и `no title`;
4. дедупликация.

Итог:
- основной корпус: `1500` новостей (`data/raw/russian_financial_news_clean_1500.csv`);
- независимый holdout-корпус: `300` новостей (`data/test/test_corpus_clean_300.csv`).

### 2.3 NER-разметка и схема
Используемые типы сущностей:
- `Company`
- `Index`
- `Event`
- `Commodity`
- `Currency`
- `Person`
- `Location`
- `Sector`
- `Time`

NER-датасеты:
- train: `11278` сущностей (`data/labeled/ner_entity_dataset.csv`);
- holdout: `858` сущностей (`data/test/test_ner_dataset.csv`).

Схема строки NER:
`sample_id, news_id, published_at, source, title, context, entity, entity_type, start_char, end_char`.

### 2.4 Наборы для тональности
- train/valid: `1500` строк (`data/labeled/entity_sentiment_dataset.csv`);
- независимый test: `274` строки (`data/test/test_entity_sentiment_dataset.csv`).

---

## 3. Методология

### 3.1 NER-модель A: spaCy + EntityRuler (baseline)
Использовано:
1. `ru_core_news_sm` для базового NER;
2. `EntityRuler` со словарными финансовыми паттернами (`config/rare_company_index_patterns.json`).

Особенность baseline: модель не дообучается на наших данных, поэтому риски классического переобучения для baseline отсутствуют.

### 3.2 NER-модель B: HF ruBERT-base (fine-tuned)
Базовый чекпоинт: `DeepPavlov/rubert-base-cased`.

Постановка: token classification с BIO-тегами.

Параметры последнего прогона (`hf_rubert_base_ner_v2_cpu`):
1. `epochs = 8`
2. `batch_size = 4`
3. `learning_rate = 2e-5`
4. `weight_decay = 0.01`
5. `warmup_ratio = 0.1`
6. `max_length = 256`
7. `stride = 96`
8. `balance_train = true`
9. устройство: `CPU`

Данные обучения:
- `train_docs = 10521`
- `valid_docs = 889`

### 3.3 Анализ тональности по сущностям
Базовый sentiment-подход:
1. `TfidfVectorizer` (uni+bi grams, `max_features=50000`, `min_df=2`);
2. `LogisticRegression` (`class_weight=balanced`, `max_iter=3000`).

Также выполнено сравнение с `RuBERT-base` (sequence classification) на независимом test.

### 3.4 Метрики
Для оценки применяются:
1. `Accuracy`
2. `Precision`, `Recall`, `F1-score`
3. `Macro F1`
4. `Weighted F1`

---

## 4. Результаты

### 4.1 NER: агрегированные метрики на holdout (858)

| Модель | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |
|---|---:|---:|---:|---:|---:|
| spaCy + EntityRuler | 0.7179 | 0.5988 | 0.5772 | 0.5872 | 0.7287 |
| HF ruBERT-base (fine-tuned) | 0.9627 | 0.8975 | 0.8503 | 0.8727 | 0.9772 |

Прирост HF ruBERT-base относительно baseline:
1. `+0.2448` по Accuracy
2. `+0.2854` по Macro F1
3. `+0.2485` по Weighted F1

Дополнительно (без служебного класса `None`, support=0):
1. spaCy macro F1: `0.6525`
2. ruBERT macro F1: `0.9696`

### 4.2 NER: поклассовые F1

| Класс | spaCy + EntityRuler | HF ruBERT-base |
|---|---:|---:|
| Company | 0.9969 | 0.9800 |
| Index | 1.0000 | 1.0000 |
| Event | 0.9163 | 0.9909 |
| Commodity | 1.0000 | 0.9355 |
| Currency | 0.0000 | 0.9701 |
| Person | 0.0000 | 0.9388 |
| Location | 0.9908 | 0.9813 |
| Sector | 0.9684 | 0.9348 |
| Time | 0.0000 | 0.9953 |

Ключевое наблюдение: baseline практически не покрывает `Currency`, `Person`, `Time` (F1=0), тогда как fine-tuned ruBERT покрывает все 9 классов.

### 4.3 Анализ ошибок NER
Основные ошибки baseline (`242/858`):
1. `Time -> None`: `108`
2. `Currency -> None`: `86`
3. `Person -> None`: `26`

Основные ошибки ruBERT (`32/858`):
1. пропуски `Company -> None`: `7`
2. пропуски `Currency -> None`: `5`
3. пропуски `Sector -> None`: `5`
4. редкие смешения `Location -> Company`: `3`

Вывод: у ruBERT ошибки преимущественно в виде пропусков, а не неверной переклассификации.

### 4.4 Диагностика переобучения по логам обучения ruBERT
Источник логов: `data/experiments/hf_rubert_base_ner_v2_cpu/hf_model/checkpoint-21048/trainer_state.json`.

Динамика:
1. train loss по эпохам: `0.3209 -> 0.0200 -> ... -> 0.0003` (монотонное снижение);
2. eval F1: `0.9627` (эп.1) -> `0.9779` (эп.8);
3. eval accuracy: `0.9820` (эп.1) -> `0.9869` (эп.8);
4. eval loss: `0.0969` (эп.1) -> `0.1434` (эп.8), с общим трендом роста.

Интерпретация:
1. признаков катастрофического переобучения не выявлено, так как `eval_f1` и `eval_accuracy` не деградируют;
2. рост `eval_loss` при почти нулевом train loss указывает на рост уверенности модели (возможная перекалибровка вероятностей на поздних эпохах);
3. для более строгого контроля рекомендуется `early stopping` (ориентир: эпохи 6-7) и регулярная валидация на полностью ручном gold-holdout.

### 4.5 Тональность по сущностям
`TF-IDF + Logistic Regression` (valid=300):
1. `accuracy = 0.8867`
2. `macro F1 = 0.8394`
3. `weighted F1 = 0.8860`

Сравнение на независимом test (274):

| Модель | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.8650 | 0.7065 | 0.5986 | 0.6374 | 0.8516 |
| RuBERT-base (sequence classification) | 0.9015 | 0.8574 | 0.7468 | 0.7730 | 0.8919 |

---

## 5. Обсуждение

### 5.1 Что показало сравнение NER-моделей
1. `spaCy + EntityRuler` хорошо работает как быстрый и интерпретируемый baseline.
2. Fine-tuned ruBERT-base существенно лучше обобщает на независимом holdout.
3. Наиболее сильный эффект от дообучения заметен в классах, которые baseline системно пропускает (`Currency`, `Person`, `Time`).

### 5.2 Ограничения исследования
1. NER-разметка сформирована полуавтоматически; нужен полностью ручной gold-holdout.
2. Классы несбалансированы (например, `Index` и `Person` имеют малый support).
3. Обучение выполнялось на CPU, что ограничило масштаб гиперпараметрического поиска.

---

## 6. Заключение
Построен рабочий прототип для магистерского исследования: корпус данных, NER-система, модуль тональности и воспроизводимая оценка качества. На текущих данных fine-tuned `DeepPavlov/rubert-base-cased` является предпочтительной NER-моделью для дальнейшего развития, так как обеспечивает значительно более высокий `macro F1` и покрытие всех целевых типов сущностей на независимом holdout.

---

## 7. Воспроизводимость и артефакты
Ключевые файлы:
1. `new_project_ru_finnews/data/processed/spacy_ruler_eval_metrics.json`
2. `new_project_ru_finnews/data/processed/spacy_ruler_eval_predictions.csv`
3. `new_project_ru_finnews/data/experiments/hf_rubert_base_ner_v2_cpu/hf_train_metrics.json`
4. `new_project_ru_finnews/data/experiments/hf_rubert_base_ner_v2_cpu/hf_holdout_metrics.json`
5. `new_project_ru_finnews/data/experiments/hf_rubert_base_ner_v2_cpu/hf_model/checkpoint-21048/trainer_state.json`
6. `new_project_ru_finnews/data/experiments/hf_rubert_base_ner_v2_cpu/hf_training_summary.json`
7. `new_project_ru_finnews/data/processed/sentiment_tfidf_metrics.json`
8. `new_project_ru_finnews/data/experiments/sentiment_tfidf_rubert/sentiment_comparison_summary.json`

---

## 8. Дальнейшая работа
1. Подготовить полностью ручной NER gold-holdout и пересчитать метрики.
2. Ввести `early stopping` и сравнить 6/7/8 эпох на том же holdout.
3. Добавить калибровку вероятностей для ruBERT (temperature scaling).
4. Расширить словарь `EntityRuler` как production-fallback поверх ruBERT.
