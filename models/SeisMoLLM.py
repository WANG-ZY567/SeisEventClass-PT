from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo.config
from einops import rearrange
import transformers.models.gpt2 as GPT2
from peft import get_peft_model, LoraConfig
from functools import partial

from .evt6_spec_frontend import Evt6WaveformToSpecStem
from ._factory import register_model

torch._dynamo.config.cache_size_limit = 1024


def lora_setting(target_modules, r=16, lora_alpha=16, lora_dropout=0.1, bias="lora_only"):
    return LoraConfig(target_modules=target_modules, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=bias)


GPT2_lora = lora_setting("all-linear")


def _resolve_gpt2_snapshot_path() -> str:
    """
    Resolve local GPT-2 snapshot path.
    Priority:
    1) env var: SEISMOLLM_GPT2_PATH
    2) file: gpt2_model_path.txt (repo root)
    3) default relative snapshot path
    """
    import os
    from pathlib import Path

    env_path = os.environ.get("SEISMOLLM_GPT2_PATH", "").strip()
    if env_path:
        return env_path

    repo_root = Path(__file__).resolve().parent.parent
    txt = repo_root / "gpt2_model_path.txt"
    if txt.exists():
        p = txt.read_text(encoding="utf-8").strip().replace("\\", "/")
        if p:
            return p

    return "./gpt2_cache/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"


def _auto_pad_1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    dim: int = -1,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Auto pad for conv layer.
    The output of conv-layer has the shape as `ceil(x.size(dim)/stride)`.
    Use this function to replace `padding='same'` which `torch.jit` and `torch.onnx` do not support.
    """

    assert (
        kernel_size >= stride
    ), f"`kernel_size` must be greater than or equal to `stride`, got {kernel_size}, {stride}"
    pos_dim = dim if dim >= 0 else x.dim() + dim
    pds = (stride - (x.size(dim) % stride)) % stride + kernel_size - stride
    padding = (0, 0) * (x.dim() - pos_dim - 1) + (pds // 2, pds - pds // 2)
    padded_x = F.pad(x, padding, "constant", padding_value)
    return padded_x


class ScaledActivation(nn.Module):
    def __init__(self, act_layer: nn.Module, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor
        self.act = act_layer()

    def forward(self, x):
        return self.act(x) * self.scale_factor


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, act_layer, norm_layer):
        super().__init__()

        self.in_proj = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False
        )

        self.conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size,
                              stride=stride, bias=False)
        self.norm = norm_layer(out_dim)
        self.act = act_layer()

    def forward(self, x):
        x = self.in_proj(x)
        x = _auto_pad_1d(x, self.conv.kernel_size[0], self.conv.stride[0])
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Multi_Scale_Conv_Block(nn.Module):
    def __init__(
        self, scale_num, scale_stride, in_dim, out_dim, kernel_size, stride, act_layer, norm_layer
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                ConvBlock(
                    in_dim,
                    out_dim,
                    kernel_size + int(scale_stride * scale),
                    stride,
                    act_layer,
                    norm_layer,
                )
                for scale in range(scale_num)
            ]
        )

        self.out_proj = nn.Conv1d(
            in_channels=scale_num * out_dim, out_channels=out_dim, kernel_size=1, bias=False
        )
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        outs = list()
        for conv in self.convs:
            xi = conv(x)
            outs.append(xi)
        x = torch.cat(outs, dim=1)
        x = self.out_proj(x)
        x = self.norm(x)
        return x


class LLM_Block(nn.Module):
    def __init__(self, start_layer, end_layer, patch_size, lora_config, pretrain=True, freeze=True):
        super(LLM_Block, self).__init__()

        self.pretrain = pretrain
        self.freeze = freeze
        self.lora_config = lora_config
        self.patch_size = patch_size

        if pretrain:
            gpt2_path = _resolve_gpt2_snapshot_path()
            self.llm = GPT2.GPT2Model.from_pretrained(
                gpt2_path,
                output_hidden_states=True,
                local_files_only=True
            )
        else:
            print("------------------no pretrain------------------")
            self.llm = GPT2.GPT2Model(GPT2.configuration_gpt2.GPT2Config())
        self.llm.h = self.llm.h[start_layer: end_layer]

        # Random-init stack: no PEFT / no LoRA checkpoint shape — train full weights.
        if self.freeze and self.pretrain:
            self.llm = get_peft_model(self.llm, self.lora_config)
            for name, param in self.llm.named_parameters():
                if "ln" in name or "wpe" in name or "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for name, param in self.llm.named_parameters():
                if "wte" in name:
                    param.requires_grad = False

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.patch_size)
        x = rearrange(x, 'b c n p -> b n (c p)')
        x = self.llm(inputs_embeds=x).last_hidden_state
        x = rearrange(x, 'b n (c p) -> b c (n p)', p=self.patch_size)
        return x


class HeadDetectionPicking(nn.Module):
    def __init__(
        self,
        feature_channels,
        layer_channels,
        layer_kernel_sizes,
        act_layer,
        norm_layer,
        out_act_layer=nn.Identity,
        out_channels=1,
        **kwargs,
    ):
        super().__init__()

        assert len(layer_channels) == len(layer_kernel_sizes)
        self.depth = len(layer_channels)
        self.up_layers = nn.ModuleList()

        for inc, outc, kers in zip(
            [feature_channels] + layer_channels[:-1],
            layer_channels[:-1] + [out_channels * 2],
            layer_kernel_sizes,
        ):
            conv = nn.Conv1d(in_channels=inc, out_channels=outc, kernel_size=kers)
            norm = norm_layer(outc)
            act = act_layer()
            self.up_layers.append(
                nn.Sequential(OrderedDict([("conv", conv), ("norm", norm), ("act", act)]))
            )

        self.out_conv = nn.Conv1d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=7,
            padding=3,
        )
        self.out_act = out_act_layer()

    def _upsampling_sizes(self, in_size: int, out_size: int):
        sizes = [out_size] * self.depth
        factor = (out_size / in_size) ** (1 / self.depth)
        for i in range(self.depth - 2, -1, -1):
            sizes[i] = int(sizes[i + 1] / factor)
        return sizes

    def forward(self, x, x0):
        _, _, L = x.size()
        up_sizes = self._upsampling_sizes(in_size=L, out_size=x0.size(-1))
        for i, layer in enumerate(self.up_layers):
            upsize = up_sizes[i]
            x = F.interpolate(x, size=upsize, mode="linear")
            x = _auto_pad_1d(x, layer.conv.kernel_size[0], layer.conv.stride[0])
            x = layer(x)

        x = self.out_conv(x)
        x = self.out_act(x)
        return x


class HeadClassification(nn.Module):
    def __init__(self, feature_channels, num_classes, out_act_layer, **kwargs):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(feature_channels, feature_channels, 16, 4) for _ in range(2)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        self.lin = nn.Linear(feature_channels, num_classes)
        self.out_act = out_act_layer()

    def forward(self, x, _: torch.Tensor = None):
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin(x)
        x = self.out_act(x)
        return x


class HeadClassificationAttnPool(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        num_classes: int,
        out_act_layer,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.attn_logits = nn.Conv1d(feature_channels, 1, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(p=float(dropout))
        self.ln = nn.LayerNorm(feature_channels)
        self.lin = nn.Linear(feature_channels, num_classes)
        self.out_act = out_act_layer()

    def forward(self, x, _: torch.Tensor = None):
        w = self.attn_logits(x)
        w = torch.softmax(w, dim=-1)
        pooled = (x * w).sum(dim=-1)
        pooled = self.ln(pooled)
        pooled = self.dropout(pooled)
        y = self.lin(pooled)
        y = self.out_act(y)
        return y


class HeadHierEvt6Sp(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        out_act_layer,
        coarse_dropout: float = 0.3,
        fine_dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.coarse = HeadClassificationAttnPool(
            feature_channels=feature_channels,
            num_classes=2,
            out_act_layer=out_act_layer,
            dropout=coarse_dropout,
        )
        self.fine = HeadClassificationAttnPool(
            feature_channels=feature_channels,
            num_classes=6,
            out_act_layer=out_act_layer,
            dropout=fine_dropout,
        )

    def forward(self, x, x0: torch.Tensor = None):
        y_coarse = self.coarse(x, x0)
        y_fine = self.fine(x, x0)
        return (y_coarse, y_fine)


class HeadEvt6MultiAux(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        out_act_layer,
        main_dropout: float = 0.3,
        aux_dropout: float = 0.3,
        enable_se_vs_ot: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.main = HeadClassificationAttnPool(
            feature_channels=feature_channels,
            num_classes=6,
            out_act_layer=out_act_layer,
            dropout=main_dropout,
        )
        self.aux_sp = HeadClassificationAttnPool(
            feature_channels=feature_channels,
            num_classes=2,
            out_act_layer=out_act_layer,
            dropout=aux_dropout,
        )
        self.aux_seot = HeadClassificationAttnPool(
            feature_channels=feature_channels,
            num_classes=2,
            out_act_layer=out_act_layer,
            dropout=aux_dropout,
        )
        self.enable_se_vs_ot = bool(enable_se_vs_ot)
        if self.enable_se_vs_ot:
            self.aux_se_vs_ot = HeadClassificationAttnPool(
                feature_channels=feature_channels,
                num_classes=2,
                out_act_layer=out_act_layer,
                dropout=aux_dropout,
            )
        else:
            self.aux_se_vs_ot = None

    def forward(self, x, x0: torch.Tensor = None):
        y_evt6 = self.main(x, x0)
        y_sp = self.aux_sp(x, x0)
        y_seot = self.aux_seot(x, x0)
        if self.enable_se_vs_ot and self.aux_se_vs_ot is not None:
            y_se_vs_ot = self.aux_se_vs_ot(x, x0)
        else:
            y_se_vs_ot = torch.zeros(
                y_evt6.size(0),
                2,
                device=y_evt6.device,
                dtype=y_evt6.dtype,
            )
        return (y_evt6, y_sp, y_seot, y_se_vs_ot)


class HeadRegression(nn.Module):
    def __init__(self, feature_channels, out_act_layer, **kwargs):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(feature_channels, feature_channels, 16, 4) for _ in range(2)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        self.lin = nn.Linear(feature_channels, 1)
        self.out_act = out_act_layer()

    def forward(self, x, _: torch.Tensor = None):
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin(x)
        x = self.out_act(x)
        return x


class HeadBAZ(nn.Module):
    def __init__(self, feature_channels, out_act_layer, **kwargs):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(feature_channels, feature_channels, 16, 4) for _ in range(2)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        self.lin = nn.Linear(feature_channels, 2)
        self.out_act = out_act_layer()

    def forward(self, x, _: torch.Tensor = None):
        for conv in self.convs:
            x = conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin(x)
        x = self.out_act(x)
        return x[:, :1], x[:, 1:]


class SeisMoLLM(nn.Module):
    def __init__(
        self,
        in_channels=3,
        conv_scale_num=4,
        conv_scale_strides=[8, 6, 4, 2],
        conv_channels=[16, 48, 96],
        conv_kernel_sizes=[16, 8, 6, 1],
        conv_strides=[2, 2, 2, 1],
        llm_layers=3,
        d_model=768,
        patch_size=8,
        dp_head_channels=[128, 160, 192, 224],
        path_drop_rate=0.2,
        mlp_drop_rate=0.2,
        mlp_ratio=4,
        mlp_bias=True,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm1d,
        use_checkpoint=False,
        output_head=HeadRegression,
        input_downsample_factor: int = 1,
        enable_channel_weighting: bool = False,
        channel_weighting_mode: str = "softplus",
        cond_num_classes: int = 0,
        cond_embed_dropout: float = 0.0,
        in_samples: int = 8192,
        spec_input: bool = False,
        llm_pretrain: bool = True,
        llm_freeze: bool = True,
        **kwargs
    ):
        super().__init__()

        conv_scale_strides = list(conv_scale_strides)
        conv_channels = list(conv_channels)
        conv_kernel_sizes = list(conv_kernel_sizes)
        conv_strides = list(conv_strides)
        dp_head_channels = list(dp_head_channels)

        assert len(conv_channels) + 1 == len(conv_kernel_sizes) == len(conv_strides)
        conv_channels.append(d_model // patch_size)

        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size
        self.feature_channels = conv_channels[-1]
        self.input_downsample_factor = int(input_downsample_factor)
        self.spec_input = bool(spec_input)
        self.in_samples = int(in_samples)

        # Optional: learnable channel weighting for 3C waveform (Z/N/E).
        # Opt-in only to avoid affecting existing mainline experiments.
        self.enable_channel_weighting = bool(enable_channel_weighting)
        self.channel_weighting_mode = str(channel_weighting_mode).lower().strip()
        if self.enable_channel_weighting:
            if int(in_channels) != 3:
                raise ValueError(
                    f"channel weighting expects in_channels=3 (ZNE), got in_channels={in_channels}"
                )
            self._cw_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        else:
            self._cw_logits = None

        if self.input_downsample_factor not in (1, 2, 4, 8):
            raise ValueError(
                f"input_downsample_factor must be one of (1,2,4,8), got {self.input_downsample_factor}"
            )
        self._input_downsampler = (
            nn.AvgPool1d(kernel_size=self.input_downsample_factor, stride=self.input_downsample_factor)
            if self.input_downsample_factor > 1
            else nn.Identity()
        )

        if self.spec_input:
            if int(in_channels) != 3:
                raise ValueError(f"spec_input=True expects in_channels=3, got {in_channels}")
            self.convs = Evt6WaveformToSpecStem(
                in_samples=self.in_samples,
                d_model=int(d_model),
                patch_size=int(patch_size),
            )
        else:
            self.convs = nn.Sequential(
                *[
                    Multi_Scale_Conv_Block(
                        scale_num=conv_scale_num,
                        scale_stride=ss,
                        in_dim=inc,
                        out_dim=outc,
                        kernel_size=kers,
                        stride=strd,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                    )
                    for ss, inc, outc, kers, strd in zip(
                        conv_scale_strides,
                        [in_channels] + conv_channels[:-1],
                        conv_channels,
                        conv_kernel_sizes,
                        conv_strides,
                    )
                ]
            )

        self.llm_blocks = LLM_Block(
            start_layer=0,
            end_layer=llm_layers,
            patch_size=patch_size,
            lora_config=GPT2_lora,
            pretrain=bool(llm_pretrain),
            freeze=bool(llm_freeze),
        )

        # Optional: class-conditioned picking.
        # When enabled, we inject the class id embedding into the feature map
        # right after the LLM blocks (feature-wise additive modulation).
        self.cond_num_classes = int(cond_num_classes or 0)
        if self.cond_num_classes > 0:
            self.cond_embed = nn.Embedding(self.cond_num_classes, self.feature_channels)
            self.cond_dropout = nn.Dropout(p=float(cond_embed_dropout or 0.0))
        else:
            self.cond_embed = None
            self.cond_dropout = None

        if (output_head in [HeadDetectionPicking]) or (
            isinstance(output_head, partial)
            and (output_head.func in [HeadDetectionPicking])
        ):
            out_layer_channels = []
            out_layer_kernel_sizes = []
            for channel, kernel in zip(dp_head_channels, conv_kernel_sizes):
                out_layer_channels.insert(0, channel)
                out_layer_kernel_sizes.insert(0, kernel)

            self.out_head = output_head(
                in_channels=in_channels,
                feature_channels=self.feature_channels,
                layer_channels=out_layer_channels,
                layer_kernel_sizes=out_layer_kernel_sizes,
                act_layer=act_layer,
                norm_layer=norm_layer,
                path_drop_rate=path_drop_rate,
                mlp_drop_rate=mlp_drop_rate,
                mlp_ratio=mlp_ratio,
                mlp_bias=mlp_bias
            )
        else:
            self.out_head = output_head(
                feature_channels=self.feature_channels,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )

    def forward(self, x):
        # Allow conditional picking: x can be (waveform, evt6_cond).
        # For backward compatibility, keep accepting a raw waveform tensor.
        cond = None
        if isinstance(x, (tuple, list)):
            if len(x) < 1:
                raise ValueError("SeisMoLLM.forward received empty tuple/list input.")
            x_wave = x[0]
            if len(x) >= 2:
                cond = x[1]
        else:
            x_wave = x

        x_input = x_wave
        # Apply optional channel weighting on raw waveform before any other processing.
        if self._cw_logits is not None:
            if not isinstance(x_wave, torch.Tensor):
                raise TypeError("channel weighting expects waveform tensor input.")
            if x_wave.dim() != 3 or x_wave.size(1) != 3:
                raise ValueError(f"expected waveform shape [B,3,L], got {tuple(x_wave.shape)}")
            if self.channel_weighting_mode == "softplus":
                w = F.softplus(self._cw_logits)  # [3] positive weights
            elif self.channel_weighting_mode in ("linear", "none", ""):
                w = self._cw_logits
            else:
                raise ValueError(
                    f"unknown channel_weighting_mode={self.channel_weighting_mode}, expected softplus/linear"
                )
            x_wave = x_wave * w.view(1, 3, 1)
            x_input = x_wave
        if self.spec_input:
            x = self.convs(x_wave)
            x = self.llm_blocks(x)
        else:
            x = self._input_downsampler(x_wave)
            x = self.convs(x)
            x = self.llm_blocks(x)

        if cond is not None and self.cond_embed is not None:
            # cond is expected to be int class id in [0, cond_num_classes).
            if not isinstance(cond, torch.Tensor):
                cond = torch.as_tensor(cond, device=x.device)
            cond = cond.to(device=x.device, dtype=torch.long)
            emb = self.cond_embed(cond)  # [B, feature_channels]
            if self.cond_dropout is not None:
                emb = self.cond_dropout(emb)
            # Broadcast to [B, feature_channels, L]
            x = x + emb.unsqueeze(-1)

        x = self.out_head(x, x_input)
        return x

    @torch.no_grad()
    def get_channel_weights(self) -> dict:
        """Return current (interpretable) channel weights for Z/N/E."""
        if self._cw_logits is None:
            return {}
        if self.channel_weighting_mode == "softplus":
            w = F.softplus(self._cw_logits).detach().cpu().tolist()
        else:
            w = self._cw_logits.detach().cpu().tolist()
        return {
            "mode": self.channel_weighting_mode,
            "w_z": float(w[0]),
            "w_n": float(w[1]),
            "w_e": float(w[2]),
        }


@register_model
def SeisMoLLM_dpk(**kwargs):
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(HeadDetectionPicking, out_act_layer=nn.Sigmoid, out_channels=3),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_dpk_cond(**kwargs):
    # Class-conditioned phase picking.
    # The dataset provides an extra input item `evt6_cond` (class id 0..5),
    # which we inject via an embedding into the feature map.
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(HeadDetectionPicking, out_act_layer=nn.Sigmoid, out_channels=3),
        cond_num_classes=6,
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_dpk_cond_spkbest(**kwargs):
    # Same architecture as SeisMoLLM_dpk_cond.
    # The only difference is config eval ordering (spk first),
    # so early-stop chooses checkpoint by SPK metric.
    model = SeisMoLLM_dpk_cond(**kwargs)
    return model


@register_model
def SeisMoLLM_pmp(**kwargs):
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(HeadClassification, out_act_layer=partial(nn.Softmax, dim=-1), num_classes=2),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_evt6(**kwargs):
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(HeadClassification, out_act_layer=partial(nn.Softmax, dim=-1), num_classes=6),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_evt6_cw(**kwargs):
    """
    EVT6 classifier with global learnable channel weights (Z/N/E) applied on input waveform.
    This is intended for seismic cross-app analysis of component importance.
    """
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        enable_channel_weighting=True,
        channel_weighting_mode=kwargs.pop("channel_weighting_mode", "softplus"),
        output_head=partial(HeadClassification, out_act_layer=partial(nn.Softmax, dim=-1), num_classes=6),
        **kwargs,
    )
    return model


# --- EVT6 component ablations (model-name aliases) ---
# NOTE: the actual channel selection is controlled by Config.models[*]["inputs"].
# These functions only exist to make model factory aware of the model names, so that
# users can run `--model-name SeisMoLLM_evt6_z` etc without touching mainline code.


@register_model
def SeisMoLLM_evt6_z(**kwargs):
    return SeisMoLLM_evt6(**kwargs)


@register_model
def SeisMoLLM_evt6_n(**kwargs):
    return SeisMoLLM_evt6(**kwargs)


@register_model
def SeisMoLLM_evt6_e(**kwargs):
    return SeisMoLLM_evt6(**kwargs)


@register_model
def SeisMoLLM_evt6_zne(**kwargs):
    return SeisMoLLM_evt6(**kwargs)


@register_model
def SeisMoLLM_evt6_ds2(**kwargs):
    model = SeisMoLLM(
        input_downsample_factor=2,
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(HeadClassification, out_act_layer=partial(nn.Softmax, dim=-1), num_classes=6),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_evt6_ps4(**kwargs):
    model = SeisMoLLM(
        patch_size=4,
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(HeadClassification, out_act_layer=partial(nn.Softmax, dim=-1), num_classes=6),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_evt6_ps2(**kwargs):
    model = SeisMoLLM(
        patch_size=2,
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(HeadClassification, out_act_layer=partial(nn.Softmax, dim=-1), num_classes=6),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_evt5(**kwargs):
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(HeadClassification, out_act_layer=partial(nn.Softmax, dim=-1), num_classes=5),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_evt6_attnpool(**kwargs):
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(
            HeadClassificationAttnPool,
            out_act_layer=partial(nn.Softmax, dim=-1),
            num_classes=6,
            dropout=0.3,
        ),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_evt6_coarse_only(**kwargs):
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(
            HeadClassificationAttnPool,
            out_act_layer=partial(nn.Softmax, dim=-1),
            num_classes=2,
            dropout=0.3,
        ),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_evt6_hier_sp(**kwargs):
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(
            HeadHierEvt6Sp,
            out_act_layer=partial(nn.Softmax, dim=-1),
            coarse_dropout=0.3,
            fine_dropout=0.3,
        ),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_evt6_hier_sp_phase2(**kwargs):
    return SeisMoLLM_evt6_hier_sp(**kwargs)


@register_model
def SeisMoLLM_evt6_multihead(**kwargs):
    enable_se_vs_ot = bool(kwargs.pop("enable_se_vs_ot", True))
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(
            HeadEvt6MultiAux,
            out_act_layer=partial(nn.Softmax, dim=-1),
            main_dropout=0.3,
            aux_dropout=0.3,
            enable_se_vs_ot=enable_se_vs_ot,
        ),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_evt6_multihead_w01(**kwargs):
    return SeisMoLLM_evt6_multihead(**kwargs)


@register_model
def SeisMoLLM_evt6_hier_sp_w01(**kwargs):
    return SeisMoLLM_evt6_hier_sp(**kwargs)


@register_model
def SeisMoLLM_evt6_multihead_w005(**kwargs):
    return SeisMoLLM_evt6_multihead(**kwargs)


@register_model
def SeisMoLLM_evt6_hier_sp_specgpt2(**kwargs):
    """
    Same heads/loss protocol as `SeisMoLLM_evt6_hier_sp`, but waveform is converted
    to a 3C log-magnitude spectrogram stem before the shared GPT-2 `LLM_Block`.
    """
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        spec_input=True,
        output_head=partial(
            HeadHierEvt6Sp,
            out_act_layer=partial(nn.Softmax, dim=-1),
            coarse_dropout=0.3,
            fine_dropout=0.3,
        ),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_evt6_multihead_w01_specgpt2(**kwargs):
    """Same as `SeisMoLLM_evt6_multihead_w01` with spectrogram stem before GPT-2."""
    enable_se_vs_ot = bool(kwargs.pop("enable_se_vs_ot", True))
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        spec_input=True,
        output_head=partial(
            HeadEvt6MultiAux,
            out_act_layer=partial(nn.Softmax, dim=-1),
            main_dropout=0.3,
            aux_dropout=0.3,
            enable_se_vs_ot=enable_se_vs_ot,
        ),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_evt3(**kwargs):
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        output_head=partial(HeadClassification, out_act_layer=partial(nn.Softmax, dim=-1), num_classes=3),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_evt3_ps4(**kwargs):
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        patch_size=4,
        output_head=partial(HeadClassification, out_act_layer=partial(nn.Softmax, dim=-1), num_classes=3),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_evt3_ps2(**kwargs):
    model = SeisMoLLM(
        path_drop_rate=0.3,
        attn_drop_rate=0.3,
        key_drop_rate=0.3,
        mlp_drop_rate=0.3,
        other_drop_rate=0.3,
        patch_size=2,
        output_head=partial(HeadClassification, out_act_layer=partial(nn.Softmax, dim=-1), num_classes=3),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_emg(**kwargs):
    model = SeisMoLLM(
        output_head=partial(
            HeadRegression,
            out_act_layer=partial(ScaledActivation, act_layer=nn.Sigmoid, scale_factor=8),
        ),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_baz(**kwargs):
    model = SeisMoLLM(
        output_head=partial(HeadBAZ, out_act_layer=partial(nn.Tanh)),
        **kwargs,
    )
    return model


@register_model
def SeisMoLLM_dis(**kwargs):
    model = SeisMoLLM(
        output_head=partial(
            HeadRegression,
            out_act_layer=partial(ScaledActivation, act_layer=nn.Sigmoid, scale_factor=500),
        ),
        **kwargs,
    )
    return model

