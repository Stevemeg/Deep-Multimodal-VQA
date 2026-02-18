import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from models.text_encoder import TextEncoder

encoder = TextEncoder()

questions = [
    "What color is the dog?",
    "How many people are in the image?"
]

embeddings = encoder(questions)

print("Text embedding shape:", embeddings.shape)
