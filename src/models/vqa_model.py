import torch
import torch.nn as nn

from src.models.text_encoder import TextEncoder
from src.models.fusion import DeepFusionTransformer


class VQAModel(nn.Module):
    def __init__(self, num_answers=1000):
        super().__init__()

        self.text_encoder = TextEncoder()

        self.fusion = DeepFusionTransformer(
            hidden_dim=768,
            num_heads=8,
            num_layers=3,
            dropout=0.1
        )

        self.classifier = nn.Linear(768, num_answers)

    def forward(self, image_embeddings, questions, return_attention=False):

        text_tokens = self.text_encoder(questions)

        fused_tokens, attention_maps = self.fusion(
            text_tokens,
            image_embeddings
        )

        cls_token = fused_tokens[:, 0, :]
        logits = self.classifier(cls_token)

        if return_attention:
            return logits, attention_maps

        return logits
