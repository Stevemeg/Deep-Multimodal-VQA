import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from models.fusion import CrossAttentionFusion

B = 2
text_tokens = torch.randn(B, 10, 768)
image_tokens = torch.randn(B, 50, 768)

fusion = CrossAttentionFusion()

output = fusion(text_tokens, image_tokens)

print("Fusion output shape:", output.shape)
