param(
    [Parameter(Mandatory = $true)]
    [string]$DataDir,
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [string]$ModelName = "SeisMoLLM_evt6_multihead_w005",
    [string]$LogDir = ".\logs",
    [int]$BatchSize = 1,
    [int]$Workers = 0,
    [string]$Device = "cuda:0"
)

$repoDir = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoDir ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Project environment not found: $pythonExe. Create it as described in docs/RUN_GUIDE_ZH.md."
}

Push-Location $repoDir
try {
    & $pythonExe main.py `
        --mode test `
        --model-name $ModelName `
        --dataset-name diting2_evt6 `
        --data $DataDir `
        --checkpoint $Checkpoint `
        --log-base $LogDir `
        --batch-size $BatchSize `
        --workers $Workers `
        --device $Device `
        --use-torch-compile false
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
