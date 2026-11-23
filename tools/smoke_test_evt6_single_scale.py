#!/usr/bin/env python3
"""Check the EVT6 single-scale tokenizer ablation can train one batch."""

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import create_model  # noqa: E402


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(
        model_name="SeisMoLLM_evt6_single_scale",
        in_channels=3,
        in_samples=8192,
        llm_start_layer=0,
        llm_layers=1,
        d_model=96,
        llm_pretrain=False,
        llm_freeze=False,
    ).to(device)
    scales = [len(block.convs) for block in model.convs]
    if scales != [1, 1, 1, 1]:
        raise AssertionError(f"Expected one tokenizer scale per block, got {scales}")
    x = torch.randn(2, 3, 8192, device=device)
    y = torch.tensor([0, 1], device=device)
    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, y)
    loss.backward()
    if output.shape != (2, 6) or not torch.isfinite(loss):
        raise AssertionError("Invalid model output or loss")
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        f"Single-scale EVT6 smoke test passed on {device}: "
        f"scales={scales}, output={tuple(output.shape)}, loss={loss.item():.6f}, "
        f"parameters={total}, trainable={trainable}"
    )


if __name__ == "__main__":
    main()
