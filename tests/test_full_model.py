import sys
import os
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from models.vqa_model import VQAModel

B = 2

# Fake image embeddings
image_embeddings = torch.randn(B, 50, 768)

questions = [
    "What color is the dog?",
    "How many people are there?"
]

model = VQAModel(num_answers=1000)

logits = model(image_embeddings, questions)

print("Logits shape:", logits.shape)
