param(
    [string]$Python = "python",
    [string]$Csv = "data\intraday\es\ES6.csv",
    [string]$ArtifactRoot = "artifacts\phase2\candidates",
    [string]$SummaryDir = "runs\phase2_generational",
    [string]$BaselineTag = "retrain_v4",
    [string]$DeploymentBaselineTag = "retrain_v6_pass2_grid_02",
    [string]$TagPrefix = "retrain_v7",
    [int]$Generations = 1,
    [int]$GenerationBudget = 8,
    [ValidateSet("auto", "trade_recovery", "fade_control", "balanced")]
    [string]$SearchBias = "auto",
    [string]$TrainStart = "2021-01-14",
    [string]$TrainEnd = "2024-12-31",
    [string]$ValStart = "2025-06-01",
    [string]$ValEnd = "2025-10-31",
    [string]$TestStart = "2025-11-01",
    [string]$TestEnd = "2026-01-16",
    [int]$NEstimators = 1600,
    [double]$MaxBadFadeRate = 0.18,
    [double]$MaxFlipRate = 0.75,
    [int]$MinTradesVal = 30,
    [int]$MinTradesFloor = 20,
    [switch]$FreshRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$cmd = @(
    $Python,
    "tools/train_phase2_generational.py",
    "--csv", $Csv,
    "--baseline-tag", $BaselineTag,
    "--deployment-baseline-tag", $DeploymentBaselineTag,
    "--artifact-root", $ArtifactRoot,
    "--summary-dir", $SummaryDir,
    "--tag-prefix", $TagPrefix,
    "--generations", $Generations,
    "--generation-budget", $GenerationBudget,
    "--search-bias", $SearchBias,
    "--train-start", $TrainStart,
    "--train-end", $TrainEnd,
    "--val-start", $ValStart,
    "--val-end", $ValEnd,
    "--test-start", $TestStart,
    "--test-end", $TestEnd,
    "--n-estimators", $NEstimators,
    "--max-bad-fade-rate", $MaxBadFadeRate,
    "--max-flip-rate", $MaxFlipRate,
    "--min-trades-val", $MinTradesVal,
    "--min-trades-floor", $MinTradesFloor,
    "--require-slippage-pass",
    "--execute"
)

if ($FreshRun) {
    $cmd += "--fresh-run"
}

& $cmd[0] $cmd[1..($cmd.Length - 1)]
