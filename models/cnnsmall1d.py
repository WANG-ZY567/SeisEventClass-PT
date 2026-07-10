import torch
import torch.nn as nn

from ._factory import register_model

__all__ = ["CNNsmall_evt6", "CNNsmall_evt5", "CNNsmall_evt3"]


class CNNsmall1D(nn.Module):
    """
    Lightweight 1D CNN baseline.
    Input:  [B, C, L]
    Output: [B, num_classes] probabilities (Softmax)

    Note: repository loss `models.loss.CELoss` expects probabilities (already softmax).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 6, drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 128, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=float(drop)) if drop and float(drop) > 0 else nn.Identity(),
            nn.Linear(128, int(num_classes)),
        )
        self.out = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return self.out(logits)


@register_model
def CNNsmall_evt6(**kwargs):
    """EVT6 6-way CNN baseline."""
    return CNNsmall1D(in_channels=kwargs.get("in_channels", 3), num_classes=6, drop=kwargs.get("drop", 0.0))


@register_model
def CNNsmall_evt5(**kwargs):
    """EVT5 5-way CNN baseline."""
    return CNNsmall1D(in_channels=kwargs.get("in_channels", 3), num_classes=5, drop=kwargs.get("drop", 0.0))


@register_model
def CNNsmall_evt3(**kwargs):
    """EVT3 3-way CNN baseline."""
    return CNNsmall1D(in_channels=kwargs.get("in_channels", 3), num_classes=3, drop=kwargs.get("drop", 0.0))

