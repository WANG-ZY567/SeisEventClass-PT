import torch
import torch.nn as nn
import torch.nn.functional as F

from ._factory import register_model

__all__ = [
    "InceptionTime_evt6",
    "InceptionTime_evt6_ds2",
    "InceptionTime_evt6_ds4",
    "InceptionTime_evt6_ds8",
    "InceptionTime_evt5",
    "InceptionTime_evt3",
    "InceptionTime_evt3_ds2",
    "InceptionTime_evt3_ds4",
    "InceptionTime_evt3_ds8",
]


class InceptionModule1D(nn.Module):
    """
    InceptionTime-style module (1D):
    - optional bottleneck 1x1
    - parallel convs with multiple kernel sizes
    - maxpool + 1x1 branch
    - concat + BN + ReLU
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_sizes=(9, 19, 39),
        bottleneck_ch: int | None = 32,
    ):
        super().__init__()
        ks = list(kernel_sizes)
        assert len(ks) == 3, "InceptionTime typically uses 3 kernel sizes"

        if bottleneck_ch is None or in_ch <= 1:
            self.bottleneck = None
            bch = in_ch
        else:
            bch = int(bottleneck_ch)
            self.bottleneck = nn.Conv1d(in_ch, bch, kernel_size=1, bias=False)

        # split output channels across 4 branches (3 conv + 1 pool)
        # keep it divisible; fall back to floor split
        b_out = max(int(out_ch) // 4, 1)
        conv_out = b_out
        pool_out = int(out_ch) - 3 * conv_out
        if pool_out <= 0:
            pool_out = conv_out
            out_ch = 4 * conv_out

        self.conv1 = nn.Conv1d(bch, conv_out, kernel_size=ks[0], padding=ks[0] // 2, bias=False)
        self.conv2 = nn.Conv1d(bch, conv_out, kernel_size=ks[1], padding=ks[1] // 2, bias=False)
        self.conv3 = nn.Conv1d(bch, conv_out, kernel_size=ks[2], padding=ks[2] // 2, bias=False)

        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_ch, pool_out, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm1d(conv_out * 3 + pool_out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x_in = x
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y3 = self.conv3(x)
        y4 = self.pool_conv(self.pool(x_in))

        y = torch.cat([y1, y2, y3, y4], dim=1)
        y = self.bn(y)
        y = self.act(y)
        return y


class InceptionBlock1D(nn.Module):
    """
    Stack multiple Inception modules with a residual connection every 3 modules.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        depth: int = 6,
        kernel_sizes=(9, 19, 39),
        bottleneck_ch: int | None = 32,
        drop: float = 0.1,
    ):
        super().__init__()
        self.depth = int(depth)
        self.drop = nn.Dropout(p=float(drop))

        mods = []
        ch = in_ch
        for i in range(self.depth):
            mods.append(
                InceptionModule1D(
                    in_ch=ch,
                    out_ch=out_ch,
                    kernel_sizes=kernel_sizes,
                    bottleneck_ch=bottleneck_ch,
                )
            )
            ch = out_ch
        self.mods = nn.ModuleList(mods)

        # residual projection (applies at i=2,5,...)
        self.res_proj = nn.ModuleList()
        for i in range((self.depth + 2) // 3):
            self.res_proj.append(
                nn.Sequential(
                    nn.Conv1d(in_ch if i == 0 else out_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_ch),
                )
            )
        self.res_act = nn.ReLU(inplace=True)

    def forward(self, x):
        res_idx = 0
        res = x
        for i, m in enumerate(self.mods):
            x = m(x)
            x = self.drop(x)
            if (i + 1) % 3 == 0:
                # apply residual
                res = self.res_proj[res_idx](res)
                res_idx += 1
                x = self.res_act(x + res)
                res = x
        return x


class InceptionTime1D(nn.Module):
    """
    InceptionTime-style 1D classifier.
    Output probabilities (Softmax) to be compatible with current CELoss implementation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        feat_ch: int = 128,
        depth: int = 6,
        kernel_sizes=(9, 19, 39),
        bottleneck_ch: int | None = 32,
        drop: float = 0.2,
        downsample_factor: int = 1,
    ):
        super().__init__()
        ds = int(downsample_factor)
        if ds < 1:
            raise ValueError(f"downsample_factor must be >=1, got {ds}")
        if ds == 1:
            self.stem = nn.Identity()
        else:
            # lightweight strided stem to reduce length before heavy inception blocks
            self.stem = nn.Sequential(
                nn.Conv1d(int(in_channels), int(feat_ch), kernel_size=9, stride=ds, padding=4, bias=False),
                nn.BatchNorm1d(int(feat_ch)),
                nn.ReLU(inplace=True),
            )
        self.block = InceptionBlock1D(
            in_ch=int(in_channels) if ds == 1 else int(feat_ch),
            out_ch=int(feat_ch),
            depth=int(depth),
            kernel_sizes=kernel_sizes,
            bottleneck_ch=bottleneck_ch,
            drop=drop,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head_drop = nn.Dropout(p=float(drop))
        self.fc = nn.Linear(int(feat_ch), int(num_classes))
        self.out = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.stem(x)
        x = self.block(x)
        x = self.pool(x).squeeze(-1)
        x = self.head_drop(x)
        x = self.fc(x)
        x = self.out(x)
        return x


@register_model
def InceptionTime_evt6(**kwargs):
    """
    EVT6 6-way classification baseline (non-LLM): InceptionTime1D.
    """
    model = InceptionTime1D(in_channels=kwargs.get("in_channels", 3), num_classes=6, downsample_factor=1)
    return model


@register_model
def InceptionTime_evt6_ds2(**kwargs):
    model = InceptionTime1D(in_channels=kwargs.get("in_channels", 3), num_classes=6, downsample_factor=2)
    return model


@register_model
def InceptionTime_evt6_ds4(**kwargs):
    model = InceptionTime1D(in_channels=kwargs.get("in_channels", 3), num_classes=6, downsample_factor=4)
    return model


@register_model
def InceptionTime_evt6_ds8(**kwargs):
    model = InceptionTime1D(in_channels=kwargs.get("in_channels", 3), num_classes=6, downsample_factor=8)
    return model


# =====================================================
# Open-source note: implementation detail.
# =====================================================


@register_model
def InceptionTime_evt3(**kwargs):
    """EVT3 baseline (3-way)."""
    model = InceptionTime1D(
        in_channels=3,
        num_classes=3,
        downsample_factor=1,
    )
    return model


@register_model
def InceptionTime_evt3_ds2(**kwargs):
    """EVT3 + model-internal downsamplex2 (faster)."""
    model = InceptionTime1D(
        in_channels=3,
        num_classes=3,
        downsample_factor=2,
    )
    return model


@register_model
def InceptionTime_evt3_ds4(**kwargs):
    """EVT3 + model-internal downsamplex4 (faster)."""
    model = InceptionTime1D(
        in_channels=3,
        num_classes=3,
        downsample_factor=4,
    )
    return model


@register_model
def InceptionTime_evt3_ds8(**kwargs):
    """EVT3 + model-internal downsamplex8 (faster)."""
    model = InceptionTime1D(
        in_channels=3,
        num_classes=3,
        downsample_factor=8,
    )
    return model


@register_model
def InceptionTime_evt5(**kwargs):
    """
    EVT5 5-way classification baseline (non-LLM): exclude se.
    """
    model = InceptionTime1D(in_channels=kwargs.get("in_channels", 3), num_classes=5, downsample_factor=1)
    return model



