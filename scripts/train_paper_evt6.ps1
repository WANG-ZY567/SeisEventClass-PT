param(
    [Parameter(Mandatory = $true)][string]$DataDir,
    [string]$LogDir = ".\logs\paper_evt6",
    [string]$PythonExe = "python",
    [string]$Device = "cuda:0",
    [string]$Gpt2Path = ""
)

$repoDir = Split-Path -Parent $PSScriptRoot
if ($Gpt2Path) { $env:SEISMOLLM_GPT2_PATH = $Gpt2Path }

Push-Location $repoDir
try {
    & $PythonExe main.py `
        --mode train_test `
        --model-name SeisMoLLM_evt6_multihead_w005 `
        --dataset-name diting2_evt6 `
        --data $DataDir `
        --log-base $LogDir `
        --force-logdir true `
        --device $Device `
        --llm GPT2 --pretrain true --freeze true `
        --LLM_start_layer 0 --LLM_layer_num 3 --d_model 768 `
        --seed 20261107 --in-samples 8192 `
        --batch-size 16 --workers 8 --epochs 200 --patience 30 `
        --optim Adam --weight-decay 0 --best-metric accuracy `
        --loss-name class_balanced_ce --cb-beta 0.995 `
        --cb-samples-per-class 4629,4629,4287,1440,160,2357 `
        --cb-weight-schedule fixed `
        --use-lr-scheduler true --lr-scheduler-name cyclic `
        --lr-scheduler-mode exp_range --lr-scheduler-gamma 1.0 `
        --base-lr 8e-5 --max-lr 1e-3 --min-lr 1e-6 `
        --warmup-steps 2000 --down-steps 3000 `
        --augmentation true --shift-event-rate 0.2 `
        --add-noise-rate 0.4 --add-gap-rate 0.4 `
        --drop-channel-rate 0.4 --scale-amplitude-rate 0.4 `
        --pre-emphasis-rate 0.4 --pre-emphasis-ratio 0.97 `
        --generate-noise-rate 0.05 --norm-mode std `
        --save-test-results true --save-test-probs true `
        --tta-times 1 --use-tensorboard true
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE." }
}
finally { Pop-Location }
