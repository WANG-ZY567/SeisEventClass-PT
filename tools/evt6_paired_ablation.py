"""Run w005 component ablations without changing the original model registry.

Use this entry point for both training and testing the generated checkpoints.
The environment variable EVT6_ABLATION selects full, random_lora, single_scale,
pretrained_no_lora, or pretrained_frozen. All retain the w005 configuration.
"""
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch
import torch.nn as nn
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from peft import get_peft_model
from models import _factory
import importlib

seismo = importlib.import_module("models.SeisMoLLM")
from config import Config

MODEL = "SeisMoLLM_evt6_multihead_w005"
VARIANT = os.environ.get("EVT6_ABLATION", "").strip()
OPTIONS = {"full", "random_lora", "single_scale", "pretrained_no_lora", "pretrained_frozen"}
if VARIANT not in OPTIONS:
    raise SystemExit(f"EVT6_ABLATION must be one of {sorted(OPTIONS)}, got {VARIANT!r}")

original_factory = _factory._name_to_creators[MODEL]
original_llm_block = seismo.LLM_Block


class MatchedLLMBlock(nn.Module):
    """Same GPT-2 config and trainability policy, with one component changed."""

    def __init__(self, start_layer, end_layer, patch_size, lora_config,
                 d_model=768, max_positions=1024, pretrain=True, freeze=True):
        super().__init__()
        if not pretrain or not freeze:
            raise ValueError("Matched ablations require --pretrain true --freeze true")
        if start_layer != 0 or end_layer <= 0:
            raise ValueError("Matched ablations currently require start layer 0")
        self.pretrain = pretrain
        self.freeze = freeze
        self.lora_config = lora_config
        self.patch_size = patch_size
        path = seismo._resolve_gpt2_snapshot_path()
        cfg = GPT2Config.from_pretrained(path, local_files_only=True)
        if cfg.n_embd != d_model or cfg.n_layer < end_layer:
            raise ValueError("Pretrained config is incompatible with requested architecture")
        if VARIANT == "random_lora":
            # Preserve the original checkpoint config, including vocabulary,
            # positional embeddings, dropout and attention dimensions.
            cfg.n_layer = int(end_layer)
            cfg.output_hidden_states = True
            self.llm = GPT2Model(cfg)
            self.llm = get_peft_model(self.llm, lora_config)
        else:
            self.llm = GPT2Model.from_pretrained(path, output_hidden_states=True,
                                                local_files_only=True)
            self.llm.h = self.llm.h[start_layer:end_layer]
        for name, parameter in self.llm.named_parameters():
            parameter.requires_grad = (VARIANT != "pretrained_frozen" and
                                       ("ln" in name or "wpe" in name or "lora" in name))

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        x = seismo.rearrange(x, 'b c n p -> b n (c p)')
        x = self.llm(inputs_embeds=x).last_hidden_state
        return seismo.rearrange(x, 'b n (c p) -> b c (n p)', p=self.patch_size)


if VARIANT in {"random_lora", "pretrained_no_lora", "pretrained_frozen"}:
    seismo.LLM_Block = MatchedLLMBlock


def make_ablation(**kwargs):
    if not kwargs.get("llm_pretrain", True) or not kwargs.get("llm_freeze", True):
        raise ValueError("All paired variants require pretrain=true and freeze=true")
    if VARIANT == "single_scale":
        kwargs["conv_scale_num"] = 1
    model = original_factory(**kwargs)
    if not hasattr(model.out_head, "forward"):
        raise RuntimeError("Missing w005 output head")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    llm_names = [n for n, p in model.llm_blocks.llm.named_parameters() if p.requires_grad]
    lora_names = [n for n in llm_names if "lora" in n.lower()]
    if bool(lora_names) != (VARIANT in {"full", "random_lora", "single_scale"}):
        raise RuntimeError(f"Unexpected LoRA parameter state for {VARIANT}")
    if VARIANT == "pretrained_frozen" and llm_names:
        raise RuntimeError("The fully frozen variant has trainable GPT-2 parameters")
    info = {"variant": VARIANT, "total_parameters": total,
            "trainable_parameters": trainable,
            "gpt2_checkpoint_path": seismo._resolve_gpt2_snapshot_path(),
            "active_transformer_layers": len(model.llm_blocks.llm.h),
            "gpt2_config": model.llm_blocks.llm.config.to_dict(),
            "lora_config": {"rank": seismo.GPT2_lora.r,
                            "alpha": seismo.GPT2_lora.lora_alpha,
                            "dropout": seismo.GPT2_lora.lora_dropout,
                            "target_modules": str(seismo.GPT2_lora.target_modules),
                            "bias": seismo.GPT2_lora.bias,
                            "active": bool(lora_names)},
            "trainable_gpt2_names": llm_names,
            "trainable_gpt2_shapes": {n: list(p.shape) for n, p in
                                      model.llm_blocks.llm.named_parameters()
                                      if p.requires_grad}}
    path = os.environ.get("EVT6_ABLATION_PARAMETER_FILE")
    if path:
        Path(path).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ABLATION] {VARIANT}: total={total:,}, trainable={trainable:,}, "
          f"trainable GPT-2 tensors={len(llm_names)}", flush=True)
    return model


_factory._name_to_creators[MODEL] = make_ablation

if sys.argv[1:] == ["--audit"]:
    from types import SimpleNamespace
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    torch.manual_seed(20261107)
    model = make_ablation(in_channels=3, in_samples=8192, llm_start_layer=0,
                          llm_layers=3, d_model=768, llm_pretrain=True,
                          llm_freeze=True).cuda().train()
    args = SimpleNamespace(loss_name="class_balanced_ce",
                           cb_samples_per_class="4172,4450,4233,1095,148,2381",
                           cb_beta=0.995, cb_class_multipliers="",
                           cb_weight_schedule="fixed", cb_warmup_epochs=10,
                           cb_ramp_epochs=20)
    loss_fn = Config.get_loss(MODEL, args).cuda()
    labels = torch.tensor([0, 3], device="cuda")
    targets = (F.one_hot(labels, 6).float(), F.one_hot((labels == 3).long(), 2).float(),
               F.one_hot(((labels == 4) | (labels == 5)).long(), 2).float(),
               F.one_hot((labels == 4).long(), 2).float())
    x = torch.randn(2, 3, 8192, device="cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-5)
    outs = model(x)
    if [tuple(t.shape) for t in outs] != [(2, 6), (2, 2), (2, 2), (2, 2)]:
        raise RuntimeError("Unexpected head output shapes")
    loss = loss_fn(outs, targets)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite loss")
    loss.backward()
    if not any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.convs.parameters() if p.requires_grad):
        raise RuntimeError("Tokenizer has no finite gradient")
    if VARIANT in {"full", "random_lora", "single_scale"} and not any(
        p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)
        for n, p in model.llm_blocks.llm.named_parameters() if "lora" in n.lower()
    ):
        raise RuntimeError("LoRA has no finite nonzero gradient")
    optimizer.step()
    torch.cuda.synchronize()
    print(f"[AUDIT OK] {VARIANT}: loss={loss.item():.6f}", flush=True)
else:
    # Preserve main.py argument parsing and implementation.
    sys.argv[0] = str(ROOT / "main.py")
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
