"""
Vanilla Transformer for 1D Time Series Classification (EVT6)
使用标准 PyTorch Transformer + 稳定的初始化
"""

import torch
import torch.nn as nn
import math
from ._factory import register_model


class StandardTransformer1D(nn.Module):
    """
    标准 Transformer Encoder for 时序分类
    使用 PyTorch 内置组件，确保数值稳定性
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
        downsample_factor=16,  # 降采样因子
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.d_model = d_model
        self.downsample_factor = downsample_factor
        
        # 计算下采样后的序列长度
        self.seq_len = max_seq_len // downsample_factor
        
        # 下采样 + 投影: [B, 3, 8192] -> [B, d_model, 512]
        # 不用 BatchNorm，避免数值不稳定
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=downsample_factor, stride=downsample_factor),
            nn.GELU(),
        )
        
        # 可学习的位置编码
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.seq_len, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        self.dropout = nn.Dropout(dropout)
        
        # 标准 PyTorch Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN，更稳定
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        
        # 分类头
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        # 初始化权重
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
            x: [B, 3, 8192] - 三通道波形
        
        Returns:
            probs: [B, num_classes]  (softmax 概率，匹配本仓库的 `CELoss` 约定)
        """
        B = x.shape[0]
        
        # 检查并清理输入数据
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"⚠️ 输入数据有 nan/inf!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 下采样和投影: [B, 3, 8192] -> [B, d_model, seq_len]
        x = self.stem(x)  # [B, d_model, seq_len]
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"⚠️ stem 输出有 nan/inf!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 转置为 [B, seq_len, d_model]
        x = x.transpose(1, 2)  # [B, seq_len, d_model]
        
        # 添加位置编码
        x = x + self.pos_embedding[:, :x.size(1), :]
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"⚠️ 位置编码后有 nan/inf!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # [B, seq_len, d_model]
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"⚠️ transformer_encoder 输出有 nan/inf!")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Global average pooling
        x = x.mean(dim=1)  # [B, d_model]
        
        # 分类
        logits = self.head(x)  # [B, num_classes]
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"⚠️ 最终输出有 nan/inf!")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)

        # 注意：本仓库的 `models.loss.CELoss` 期望输入是概率（已 softmax）
        probs = torch.softmax(logits, dim=1)
        return probs


# =====================================================
# EVT6 任务的具体配置
# =====================================================

@register_model
def Transformer1D_evt6(**kwargs):
    """
    EVT6 分类的标准 Transformer
    
    配置：
    - d_model=256 (较小，避免过拟合)
    - nhead=8
    - num_layers=4 (较浅，提高稳定性)
    - downsample_factor=16 (8192 -> 512，避免显存爆炸)
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
    EVT6 Transformer baseline（更长序列）：
    - downsample_factor=8  (8192 -> 1024 tokens)

    说明：序列更长，注意显存；建议 batch_size<=32（不够就降到16）。
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
    ds8 + 更大容量：
    - downsample_factor=8  (8192 -> 1024 tokens)
    - d_model=512, num_layers=6, dim_feedforward=2048

    说明：比 ds8 基线更强，通常 80G 显存可配 batch_size≈16~32（视是否启用 AMP/compile）。
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
    EVT6 Transformer baseline（更长序列）：
    - downsample_factor=4  (8192 -> 2048 tokens)

    说明：注意 self-attention 复杂度 O(L^2)；建议 batch_size<=16（不够就降到8）。
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
    ds4 + 更大容量：
    - downsample_factor=4  (8192 -> 2048 tokens)
    - d_model=512, num_layers=6, dim_feedforward=2048

    说明：注意 self-attention O(L^2)；建议 batch_size≈8~16 起步（80G 视实现/缓存而定）。
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
    EVT6 Transformer baseline（超长序列）：
    - downsample_factor=2  (8192 -> 4096 tokens)

    说明：self-attention 复杂度 O(L^2)；强烈建议 batch_size<=8（不够就降到4/2）。
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
    更小的配置（快速实验）
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
    更大的配置（如果小模型效果好）
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
# EVT3 任务的具体配置（3-way classification）
# =====================================================


@register_model
def Transformer1D_evt3(**kwargs):
    """
    EVT3 分类的标准 Transformer（默认 ds16 -> 512 tokens）。
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
    """EVT3 + 更长序列（ds8 -> 1024 tokens）。"""
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
    """EVT3 + 更长序列（ds4 -> 2048 tokens）。"""
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
    # 测试
    model = Transformer1D_evt6()
    model.eval()
    
    x = torch.randn(4, 3, 8192)
    y = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试训练
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
