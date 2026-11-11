param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset = "data\kvasir-v2",
    [string]$Output = "outputs\paper_split",
    [int]$Seed = 2023,
    [int]$TestPerClass = 200,
    [int]$TrainPerClass = 872,
    [int]$BatchSize = 32
)

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python environment not found at $python. Create it with: py -3.11 -m venv .venv"
}
if (-not (Test-Path $Dataset)) {
    throw "Dataset directory not found: $Dataset"
}

& $python (Join-Path $PSScriptRoot "prepare_dataset.py") `
    --source $Dataset `
    --output $Output `
    --test-per-class $TestPerClass `
    --train-per-class $TrainPerClass `
    --seed $Seed

& $python (Join-Path $PSScriptRoot "extract_features.py") `
    --manifest (Join-Path $Output "manifest.csv") `
    --output (Join-Path $Output "features.npz") `
    --batch-size $BatchSize

& $python (Join-Path $PSScriptRoot "evaluate_retrieval.py") `
    --features (Join-Path $Output "features.npz") `
    --output-dir (Join-Path $Output "metrics")

Write-Host "Reproduction complete. Metrics are in $(Join-Path $Output 'metrics')."
