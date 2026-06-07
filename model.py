import math
import torch
import torch.nn as nn
from config import *


class VelocityTransformer(nn.Module):
    """Encoder-only Transformer，预测每个音符的力度"""

    def __init__(self):
        super().__init__()

        # 输入投影: 4 维 → D_MODEL 维
        self.input_proj = nn.Linear(4, D_MODEL)

        # 可学习位置编码
        self.pos_embed = nn.Embedding(MAX_NOTES, D_MODEL)

        # Transformer Encoder（PyTorch 自带）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            activation='gelu',
            batch_first=True,  # (batch, seq, dim) 格式
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)

        # 分类头: D_MODEL → 128 (velocity 0-127)
        self.classifier = nn.Linear(D_MODEL, NUM_CLASSES)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, padding_mask=None):
        """
        x: (batch, seq_len, 4) — 输入特征
        padding_mask: (batch, seq_len) — True 表示是 padding，需要忽略
        返回: (batch, seq_len, 128) — 每个位置的 velocity logits
        """
        batch_size, seq_len, _ = x.shape

        # 输入投影
        x = self.input_proj(x)  # (batch, seq, d_model)

        # 位置编码
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embed(positions)

        # Transformer Encoder
        # src_key_padding_mask: True 表示该位置被 mask（即 padding）
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # 分类
        logits = self.classifier(x)  # (batch, seq, 128)

        return logits
