param(
    [Parameter(Mandatory = $true)][string]$DataDir,
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [string]$LogDir = ".\logs\paper_evt6_eval",
    [string]$PythonExe = "python",
    [string]$Device = "cuda:0",
    [string]$Gpt2Path = ""
)

$repoDir = Split-Path -Parent $PSScriptRoot
if ($Gpt2Path) { $env:SEISMOLLM_GPT2_PATH = $Gpt2Path }

Push-Location $repoDir
try {
    & $PythonExe main.py `
        --mode test `
        --model-name SeisMoLLM_evt6_multihead_w005 `
        --dataset-name diting2_evt6 `
        --data $DataDir --checkpoint $Checkpoint --checkpoint-strict true `
        --log-base $LogDir --force-logdir true --device $Device `
        --llm GPT2 --pretrain true --freeze true `
        --LLM_start_layer 0 --LLM_layer_num 3 --d_model 768 `
        --seed 20261107 --in-samples 8192 `
        --batch-size 16 --workers 8 --norm-mode std `
        --augmentation false --shuffle false `
        --save-test-results true --save-test-probs true `
        --tta-times 1 --use-tensorboard false
    if ($LASTEXITCODE -ne 0) { throw "Evaluation failed with exit code $LASTEXITCODE." }
}
finally { Pop-Location }
