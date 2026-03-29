param(
    [string]$PythonExe = "python",
    [string]$ProjectRoot = "",
    [int]$MaxRetries = 5
)

$ErrorActionPreference = "Stop"

if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
}

Write-Host "[info] Starting ruBERT-base NER training..."
Write-Host "[info] ProjectRoot: $ProjectRoot"

Set-Location $ProjectRoot

# Recommended env for flaky HF downloads on Windows.
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:TOKENIZERS_PARALLELISM = "false"

$cmd = @(
    "new_project_ru_finnews/src/ner/train_hf_rubert_ner.py",
    "--train-csv", "new_project_ru_finnews/data/labeled/ner_entity_dataset.csv",
    "--test-csv", "new_project_ru_finnews/data/test/test_ner_dataset.csv",
    "--output-dir", "new_project_ru_finnews/data/experiments/hf_rubert_base_ner",
    "--hf-model", "DeepPavlov/rubert-base-cased",
    "--epochs", "8",
    "--batch-size", "4",
    "--learning-rate", "2e-5",
    "--weight-decay", "0.01",
    "--warmup-ratio", "0.1",
    "--max-length", "256",
    "--stride", "96",
    "--balance-train"
)

for ($attempt = 1; $attempt -le $MaxRetries; $attempt++) {
    Write-Host ""
    Write-Host "[info] Attempt $attempt / $MaxRetries"
    & $PythonExe @cmd

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "[ok] Training finished successfully."
        Write-Host "[ok] Metrics: new_project_ru_finnews/data/experiments/hf_rubert_base_ner/hf_holdout_metrics.json"
        Write-Host "[ok] Summary: new_project_ru_finnews/data/experiments/hf_rubert_base_ner/hf_training_summary.json"
        exit 0
    }

    if ($attempt -lt $MaxRetries) {
        $delay = [Math]::Pow(2, $attempt)
        Write-Host "[warn] Training failed with exit code $LASTEXITCODE. Retry in $delay sec..."
        Start-Sleep -Seconds $delay
    }
}

Write-Host ""
Write-Host "[error] All retries failed."
exit 1
