import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8, dropout=0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, text_tokens, image_tokens):

        attn_output, attn_weights = self.cross_attn(
            query=text_tokens,
            key=image_tokens,
            value=image_tokens,
            need_weights=True
        )

        x = self.norm1(text_tokens + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x, attn_weights


class DeepFusionTransformer(nn.Module):
    def __init__(self, hidden_dim=768, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, text_tokens, image_tokens):

        attention_maps = []
        x = text_tokens

        for layer in self.layers:
            x, attn = layer(x, image_tokens)
            attention_maps.append(attn)

        return x, attention_maps
