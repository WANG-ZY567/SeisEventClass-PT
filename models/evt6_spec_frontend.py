"""
Waveform -> (Z/N/E log-magnitude spectrogram) -> 2D CNN stem -> [B, C, L]

Designed so that C = d_model // patch_size (default 768//8=96) and L is a
multiple of patch_size (default 8), matching `LLM_Block` unfold contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Evt6WaveformToSpecStem(nn.Module):
    """
    Args:
        in_samples: waveform length per channel (default 8192).
        d_model / patch_size: output channels = d_model // patch_size.
        out_seq_len: temporal length after pooling (default 256, divisible by 8).
    """

    def __init__(
        self,
        in_samples: int = 8192,
        d_model: int = 768,
        patch_size: int = 8,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: int = 256,
        out_seq_len: int = 256,
        log_spec: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        if out_seq_len % int(patch_size) != 0:
            raise ValueError(f"out_seq_len={out_seq_len} must be divisible by patch_size={patch_size}")
        self.in_samples = int(in_samples)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.out_seq_len = int(out_seq_len)
        self.log_spec = bool(log_spec)
        self.eps = float(eps)
        self.out_channels = int(d_model) // int(patch_size)

        self.register_buffer("_window", torch.hann_window(self.win_length, periodic=True))

        self.stem2d = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.GELU(),
        )
        self.pool_time = nn.AdaptiveAvgPool1d(self.out_seq_len)

    def _stft_mag(self, w: torch.Tensor) -> torch.Tensor:
        """w: [B, L] -> mag [B, F, T] float"""
        spec = torch.stft(
            w,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self._window.to(dtype=w.dtype, device=w.device),
            center=True,
            return_complex=True,
        )
        mag = spec.abs()
        if self.log_spec:
            mag = torch.log(mag + self.eps)
        # per-sample standardization on (F, T)
        m = mag.mean(dim=(-2, -1), keepdim=True)
        s = mag.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        mag = (mag - m) / s
        return mag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, L] waveform (same layout as SeisMoLLM waveform input).
        returns: [B, out_channels, out_seq_len]
        """
        if x.dim() != 3 or x.size(1) != 3:
            raise ValueError(f"Evt6WaveformToSpecStem expects [B,3,L], got {tuple(x.shape)}")
        b, _, L = x.shape
        if L != self.in_samples:
            # pad / crop defensively (match spectrogram baseline tolerance)
            if L < self.in_samples:
                x = F.pad(x, (0, self.in_samples - L))
            else:
                x = x[..., : self.in_samples]

        # per-trace max-norm (Z/N/E) before STFT — matches common EVT6 preprocessing
        denom = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        x = x / denom

        mags = []
        for c in range(3):
            mags.append(self._stft_mag(x[:, c, :]))
        s = torch.stack(mags, dim=1)  # [B, 3, F, T]
        s = self.stem2d(s)  # [B, C, F', T']
        s = s.mean(dim=2)  # freq pool -> [B, C, T']
        s = self.pool_time(s)  # [B, C, out_seq_len]
        return s
