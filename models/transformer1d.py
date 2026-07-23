"""
Vanilla Transformer for 1D Time Series Classification (EVT6)
data PyTorch Transformer + data
"""

import torch
import torch.nn as nn
import math
from ._factory import register_model


class StandardTransformer1D(nn.Module):
    """
    data Transformer Encoder for data
    data PyTorch data, data
    """
    def __init__(
        self,
        in_channels=3,
        num_classes=6,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=8192,
        downsample_factor=16,  # Open-source note: implementation detail.
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.d_model = d_model
        self.downsample_factor = downsample_factor
        
        # Open-source note: implementation detail.
        self.seq_len = max_seq_len // downsample_factor
        
        # Open-source note: implementation detail.
        # Open-source note: implementation detail.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=downsample_factor, stride=downsample_factor),
            nn.GELU(),
        )
        
        # Open-source note: implementation detail.
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        self.dropout = nn.Dropout(dropout)
        
        # Open-source note: implementation detail.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Open-source note: implementation detail.
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        
        # Open-source note: implementation detail.
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        # Open-source note: implementation detail.
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, 8192] - data
        
        Returns:
            probs: [B, num_classes]  (softmax data, data `CELoss` data)
        """
        B = x.shape[0]
        
        # Open-source note: implementation detail.
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[WARN] data nan/inf!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Open-source note: implementation detail.
        x = self.stem(x)  # [B, d_model, seq_len]
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[WARN] stem data nan/inf!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Open-source note: implementation detail.
        x = x.transpose(1, 2)  # [B, seq_len, d_model]
        
        # Open-source note: implementation detail.
        x = x + self.pos_embedding[:, :x.size(1), :]
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[WARN] data nan/inf!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # [B, seq_len, d_model]
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"[WARN] transformer_encoder data nan/inf!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Global average pooling
        x = x.mean(dim=1)  # [B, d_model]
        
        # Open-source note: implementation detail.
        logits = self.head(x)  # [B, num_classes]
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"[WARN] data nan/inf!")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)

        # Open-source note: implementation detail.
        probs = torch.softmax(logits, dim=1)
        return probs


# =====================================================
# Open-source note: implementation detail.
# =====================================================

@register_model
def Transformer1D_evt6(**kwargs):
    """
    EVT6 data Transformer
    
    value: 
    - d_model=256 (data, data)
    - nhead=8
    - num_layers=4 (data, data)
    - downsample_factor=16 (8192 -> 512, data)
    """
    model = StandardTransformer1D(
        in_channels=3,
        num_classes=6,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=8192,
        downsample_factor=16,
    )
    return model


@register_model
def Transformer1D_evt6_ds8(**kwargs):
    """
    EVT6 Transformer baseline(data): 
    - downsample_factor=8  (8192 -> 1024 tokens)

    value: data, data; data batch_size<=32(text16). 
    """
    model = StandardTransformer1D(
        in_channels=3,
        num_classes=6,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=8192,
        downsample_factor=8,
    )
    return model


@register_model
def Transformer1D_evt6_ds8_large(**kwargs):
    """
    ds8 + value: 
    - downsample_factor=8  (8192 -> 1024 tokens)
    - d_model=512, num_layers=6, dim_feedforward=2048

    value: data ds8 data, data 80G data batch_sizeapprox16~32(data AMP/compile). 
    """
    model = StandardTransformer1D(
        in_channels=3,
        num_classes=6,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=8192,
        downsample_factor=8,
    )
    return model


@register_model
def Transformer1D_evt6_ds4(**kwargs):
    """
    EVT6 Transformer baseline(data): 
    - downsample_factor=4  (8192 -> 2048 tokens)

    value: data self-attention data O(L^2); data batch_size<=16(text8). 
    """
    model = StandardTransformer1D(
        in_channels=3,
        num_classes=6,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=8192,
        downsample_factor=4,
    )
    return model


@register_model
def Transformer1D_evt6_ds4_large(**kwargs):
    """
    ds4 + value: 
    - downsample_factor=4  (8192 -> 2048 tokens)
    - d_model=512, num_layers=6, dim_feedforward=2048

    value: data self-attention O(L^2); data batch_sizeapprox8~16 data(80G data/data). 
    """
    model = StandardTransformer1D(
        in_channels=3,
        num_classes=6,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=8192,
        downsample_factor=4,
    )
    return model


@register_model
def Transformer1D_evt6_ds2(**kwargs):
    """
    EVT6 Transformer baseline(data): 
    - downsample_factor=2  (8192 -> 4096 tokens)

    value: self-attention data O(L^2); data batch_size<=8(text4/2). 
    """
    model = StandardTransformer1D(
        in_channels=3,
        num_classes=6,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=8192,
        downsample_factor=2,
    )
    return model


@register_model
def Transformer1D_evt6_small(**kwargs):
    """
    data(data)
    """
    model = StandardTransformer1D(
        in_channels=3,
        num_classes=6,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        max_seq_len=8192,
        downsample_factor=16,
    )
    return model


@register_model
def Transformer1D_evt6_large(**kwargs):
    """
    data(data)
    """
    model = StandardTransformer1D(
        in_channels=3,
        num_classes=6,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=8192,
        downsample_factor=16,
    )
    return model


# =====================================================
# Open-source note: implementation detail.
# =====================================================


@register_model
def Transformer1D_evt3(**kwargs):
    """
    EVT3 data Transformer(data ds16 -> 512 tokens). 
    """
    model = StandardTransformer1D(
        in_channels=3,
        num_classes=3,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=8192,
        downsample_factor=16,
    )
    return model


@register_model
def Transformer1D_evt3_ds8(**kwargs):
    """Open-source note: implementation detail."""
    model = StandardTransformer1D(
        in_channels=3,
        num_classes=3,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=8192,
        downsample_factor=8,
    )
    return model


@register_model
def Transformer1D_evt3_ds4(**kwargs):
    """Open-source note: implementation detail."""
    model = StandardTransformer1D(
        in_channels=3,
        num_classes=3,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=8192,
        downsample_factor=4,
    )
    return model


if __name__ == "__main__":
    # Open-source note: implementation detail.
    model = Transformer1D_evt6()
    model.eval()
    
    x = torch.randn(4, 3, 8192)
    y = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Open-source note: implementation detail.
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    labels = torch.randint(0, 6, (4,))
    
    for i in range(5):
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print(f"Step {i+1}: Loss = {loss.item():.4f}")
