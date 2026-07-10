import torch
import torch.nn as nn
import torch.nn.functional as F

from ._factory import register_model

__all__ = [
    "TCN_evt6",
    "TCN_evt6_ds2",
    "TCN_evt6_ds4",
    "TCN_evt6_ds8",
    "TCN_evt6_small",
    "TCN_evt6_large",
    "TCN_evt3",
    "TCN_evt3_ds2",
    "TCN_evt3_ds4",
    "TCN_evt3_ds8",
]


def _causal_pad_1d(x: torch.Tensor, kernel_size: int, stride: int, dilation: int) -> torch.Tensor:
    """
    Causal padding for 1D conv so that output at time t only depends on <=t.
    This is consistent with the TemporalConvLayer implementation used in other models.
    """
    if stride != 1:
        # For our usage, stride is 1 inside TCN blocks.
        # Keep behavior sane if someone changes it later.
        left = (kernel_size - 1) * dilation
    else:
        left = (kernel_size - 1) * dilation
    return F.pad(x, (left, 0))


class _TCNResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, drop_rate: float):
        super().__init__()
        self.conv0 = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.bn0 = nn.BatchNorm1d(channels)
        self.relu0 = nn.ReLU(inplace=True)
        self.drop0 = nn.Dropout1d(drop_rate)

        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout1d(drop_rate)

        # residual (same channels here, so identity is enough)
        self.skip = nn.Identity()

    def forward(self, x: torch.Tensor):
        residual = self.skip(x)
        y = _causal_pad_1d(x, self.conv0.kernel_size[0], self.conv0.stride[0], self.conv0.dilation[0])
        y = self.conv0(y)
        y = self.bn0(y)
        y = self.relu0(y)
        y = self.drop0(y)

        y = _causal_pad_1d(y, self.conv1.kernel_size[0], self.conv1.stride[0], self.conv1.dilation[0])
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.drop1(y)

        # residual
        return y + residual


class TemporalConvNet1D(nn.Module):
    """
    A simple TCN encoder:
    - 1x1 projection to `channels`
    - multiple dilated causal conv residual blocks
    - returns a sequence feature map [B, C, L]
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 128,
        kernel_size: int = 6,
        dilations=None,
        num_blocks: int = 1,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        if dilations is None:
            dilations = [2**i for i in range(10)]  # 1..512 receptive field growth

        self.conv_in = nn.Conv1d(in_channels, channels, kernel_size=1)
        blocks = []
        for d in list(dilations) * int(num_blocks):
            blocks.append(_TCNResBlock(channels=channels, kernel_size=int(kernel_size), dilation=int(d), drop_rate=float(drop_rate)))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for b in self.blocks:
            x = b(x)
        return x


class TCNClassifier1D(nn.Module):
    """
    TCN baseline for EVT6 classification.
    Output probabilities (Softmax) to be compatible with current CELoss implementation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        channels: int = 128,
        kernel_size: int = 6,
        dilations=None,
        num_blocks: int = 1,
        drop_rate: float = 0.1,
        downsample_factor: int = 1,
    ):
        super().__init__()
        ds = int(downsample_factor)
        if ds < 1:
            raise ValueError(f"downsample_factor must be >=1, got {ds}")
        # simple strided conv stem to reduce sequence length and speed up GPU
        # keep it light (depthwise) so it doesn't dominate compute
        if ds == 1:
            self.stem = nn.Identity()
        else:
            self.stem = nn.Sequential(
                nn.Conv1d(
                    int(in_channels),
                    int(in_channels),
                    kernel_size=9,
                    stride=ds,
                    padding=4,
                    groups=int(in_channels),
                    bias=False,
                ),
                nn.BatchNorm1d(int(in_channels)),
                nn.ReLU(inplace=True),
            )
        self.tcn = TemporalConvNet1D(
            in_channels=in_channels,
            channels=channels,
            kernel_size=kernel_size,
            dilations=dilations,
            num_blocks=num_blocks,
            drop_rate=drop_rate,
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(p=float(drop_rate))
        self.fc = nn.Linear(int(channels), int(num_classes))
        self.out = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)             # [B, Cin, L/ds]
        x = self.tcn(x)              # [B, C, L]
        x = self.pool(x).squeeze(-1) # [B, C]
        x = self.drop(x)
        x = self.fc(x)               # logits
        x = self.out(x)              # probs for CELoss
        return x


@register_model
def TCN_evt6(**kwargs):
    model = TCNClassifier1D(
        in_channels=kwargs.get("in_channels", 3),
        num_classes=6,
        channels=256,
        kernel_size=6,
        dilations=[2**i for i in range(11)],  # up to 1024
        num_blocks=1,
        drop_rate=0.1,
        downsample_factor=1,
    )
    return model


@register_model
def TCN_evt6_ds2(**kwargs):
    model = TCNClassifier1D(
        in_channels=kwargs.get("in_channels", 3),
        num_classes=6,
        channels=256,
        kernel_size=6,
        dilations=[2**i for i in range(11)],
        num_blocks=1,
        drop_rate=0.1,
        downsample_factor=2,
    )
    return model


@register_model
def TCN_evt6_ds4(**kwargs):
    model = TCNClassifier1D(
        in_channels=kwargs.get("in_channels", 3),
        num_classes=6,
        channels=256,
        kernel_size=6,
        dilations=[2**i for i in range(11)],
        num_blocks=1,
        drop_rate=0.1,
        downsample_factor=4,
    )
    return model


@register_model
def TCN_evt6_ds8(**kwargs):
    model = TCNClassifier1D(
        in_channels=kwargs.get("in_channels", 3),
        num_classes=6,
        channels=256,
        kernel_size=6,
        dilations=[2**i for i in range(11)],
        num_blocks=1,
        drop_rate=0.1,
        downsample_factor=8,
    )
    return model


@register_model
def TCN_evt6_small(**kwargs):
    model = TCNClassifier1D(
        in_channels=kwargs.get("in_channels", 3),
        num_classes=6,
        channels=128,
        kernel_size=6,
        dilations=[2**i for i in range(9)],   # up to 256
        num_blocks=1,
        drop_rate=0.1,
        downsample_factor=1,
    )
    return model


@register_model
def TCN_evt6_large(**kwargs):
    model = TCNClassifier1D(
        in_channels=kwargs.get("in_channels", 3),
        num_classes=6,
        channels=384,
        kernel_size=6,
        dilations=[2**i for i in range(12)],  # up to 2048
        num_blocks=2,
        drop_rate=0.15,
        downsample_factor=1,
    )
    return model


# =====================================================
# EVT3 任务（3-way classification）
# =====================================================


@register_model
def TCN_evt3(**kwargs):
    """EVT3 baseline (3-way)."""
    model = TCNClassifier1D(
        in_channels=3,
        num_classes=3,
        channels=128,
        kernel_size=6,
        dilations=None,
        num_blocks=1,
        drop_rate=0.1,
        downsample_factor=1,
    )
    return model


@register_model
def TCN_evt3_ds2(**kwargs):
    """EVT3 + model-internal downsample×2 (faster)."""
    model = TCNClassifier1D(
        in_channels=3,
        num_classes=3,
        channels=128,
        kernel_size=6,
        dilations=None,
        num_blocks=1,
        drop_rate=0.1,
        downsample_factor=2,
    )
    return model


@register_model
def TCN_evt3_ds4(**kwargs):
    """EVT3 + model-internal downsample×4 (faster)."""
    model = TCNClassifier1D(
        in_channels=3,
        num_classes=3,
        channels=128,
        kernel_size=6,
        dilations=None,
        num_blocks=1,
        drop_rate=0.1,
        downsample_factor=4,
    )
    return model


@register_model
def TCN_evt3_ds8(**kwargs):
    """EVT3 + model-internal downsample×8 (faster)."""
    model = TCNClassifier1D(
        in_channels=3,
        num_classes=3,
        channels=128,
        kernel_size=6,
        dilations=None,
        num_blocks=1,
        drop_rate=0.1,
        downsample_factor=8,
    )
    return model

