# Reproducibility Notes

## Environment

Install Python dependencies:

```bash
pip install -r requirements.txt
```

For CUDA/PyTorch, choose the build that matches your GPU driver and cluster environment.

## GPT-2 Weights

SeisMoLLM adapts GPT-2 weights. Download GPT-2 with one of the helper scripts:

```bash
python download_gpt2_simple.py
```

or

```bash
python download_gpt2_modelscope.py
```

Then make sure the model path used by the code points to your local GPT-2 directory.

## Example Commands

Train and evaluate EVT6:

```bash
python main.py \
  --mode train_test \
  --model-name SeisMoLLM_evt6_multihead_w01 \
  --dataset-name diting2_evt6 \
  --data /path/to/diting2_evt6 \
  --batch-size 64 \
  --epochs 100 \
  --log-base ./logs
```

Evaluate from a checkpoint:

```bash
python main.py \
  --mode test \
  --model-name SeisMoLLM_evt6_multihead_w01 \
  --dataset-name diting2_evt6 \
  --data /path/to/diting2_evt6 \
  --checkpoint /path/to/model.pth
```

## Notes

The historical experiment folders from the working directory are intentionally not included. Keep only the scripts and documents needed to reproduce the final experiments.
