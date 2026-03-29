param(
    [string]$PythonExe = "python",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
}

Write-Host "[info] Starting spaCy transformer+ner pipeline..."
Write-Host "[info] ProjectRoot: $ProjectRoot"

Set-Location $ProjectRoot

& $PythonExe "new_project_ru_finnews/src/ner/prepare_spacy_ner_data.py" `
  --train-csv "new_project_ru_finnews/data/labeled/ner_entity_dataset.csv" `
  --holdout-csv "new_project_ru_finnews/data/test/test_ner_dataset.csv" `
  --output-dir "new_project_ru_finnews/data/spacy" `
  --dev-ratio 0.2 `
  --seed 42
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe "new_project_ru_finnews/src/ner/train_spacy_transformer_ner.py" `
  --train-spacy "new_project_ru_finnews/data/spacy/train.spacy" `
  --dev-spacy "new_project_ru_finnews/data/spacy/dev.spacy" `
  --output-dir "new_project_ru_finnews/data/experiments/spacy_transformer_ner" `
  --config-path "new_project_ru_finnews/data/experiments/spacy_transformer_ner/config.cfg" `
  --hf-transformer "DeepPavlov/rubert-base-cased" `
  --gpu-id -1 `
  --max-steps 12000 `
  --eval-frequency 250 `
  --dropout 0.15 `
  --patience 1200 `
  --seed 42 `
  --run-debug-data
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe "new_project_ru_finnews/src/ner/evaluate_spacy_finetuned_ner.py" `
  --model-dir "new_project_ru_finnews/data/experiments/spacy_transformer_ner/model-best" `
  --eval-csv "new_project_ru_finnews/data/test/test_ner_dataset.csv" `
  --output-dir "new_project_ru_finnews/data/experiments/spacy_transformer_ner"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[ok] spaCy transformer+ner pipeline completed."
