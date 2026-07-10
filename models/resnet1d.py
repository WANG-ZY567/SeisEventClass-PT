import torch
import torch.nn as nn
import torch.nn.functional as F
from ._factory import register_model

__all__ = ["ResNet1D_evt6", "ResNet1D_evt5", "ResNet1D_evt3"]


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, drop: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=float(drop))
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = out + identity
        out = self.act(out)
        return out


class ResNet1D(nn.Module):
    """
    Simple 1D-ResNet for waveform classification.
    Input:  [B, C, L]
    Output: [B, num_classes] probabilities (Softmax)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        base_channels: int = 64,
        blocks=(2, 2, 2, 2),
        drop: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        ch = base_channels
        self.layer1 = self._make_layer(ch, ch, blocks[0], stride=1, drop=drop)
        self.layer2 = self._make_layer(ch, ch * 2, blocks[1], stride=2, drop=drop)
        ch *= 2
        self.layer3 = self._make_layer(ch, ch * 2, blocks[2], stride=2, drop=drop)
        ch *= 2
        self.layer4 = self._make_layer(ch, ch * 2, blocks[3], stride=2, drop=drop)
        ch *= 2

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=float(drop))
        self.fc = nn.Linear(ch, int(num_classes))
        self.out = nn.Softmax(dim=-1)

    def _make_layer(self, in_ch: int, out_ch: int, nblocks: int, stride: int, drop: float):
        layers = [BasicBlock1D(in_ch, out_ch, stride=stride, drop=drop)]
        for _ in range(1, int(nblocks)):
            layers.append(BasicBlock1D(out_ch, out_ch, stride=1, drop=drop))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.out(x)
        return x


@register_model
def ResNet1D_evt6(**kwargs):
    """
    EVT6 6-way classification baseline (non-LLM).
    """
    model = ResNet1D(in_channels=kwargs.get("in_channels", 3), num_classes=6)
    return model


@register_model
def ResNet1D_evt5(**kwargs):
    """
    EVT5 5-way classification baseline (non-LLM): exclude se.
    """
    model = ResNet1D(in_channels=kwargs.get("in_channels", 3), num_classes=5)
    return model


@register_model
def ResNet1D_evt3(**kwargs):
    """
    EVT3 3-way classification baseline (non-LLM).
    """
    model = ResNet1D(in_channels=kwargs.get("in_channels", 3), num_classes=3)
    return model



