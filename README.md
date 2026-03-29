# News Project

Актуальная версия проекта находится в папке `new_project_ru_finnews/`.

## Быстрый старт (с нуля)

```bash
python -m pip install -r requirements.txt
cd new_project_ru_finnews
```

Запуск UI:

```bash
streamlit run src/ui/app.py
```

Запуск обучения NER (spaCy transformer+ner, CPU):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_train_spacy_transformer_ner.ps1
```

Запуск обучения NER (HF ruBERT-base, CPU):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_train_rubert_base.ps1
```

Подробная инструкция по проекту, данным, скриптам и артефактам:
- `new_project_ru_finnews/README.md`
- `new_project_ru_finnews/docs/PROJECT_REPORT.md`
- `new_project_ru_finnews/docs/SCIENTIFIC_ARTICLE_DRAFT.md`
