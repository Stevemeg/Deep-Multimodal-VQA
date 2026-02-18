import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from vqa_dataset import VQADataset

IMAGE_ROOT = r"C:\Users\konab\OneDrive\Desktop\Career\projects\VQA_Project\data\images"
EMBED_ROOT = r"C:\Users\konab\OneDrive\Desktop\Career\projects\VQA_Project\data\embeddings"

VAL_Q = r"C:\Users\konab\OneDrive\Desktop\Career\projects\VQA_Project\data\vqa\v2_OpenEnded_mscoco_val2014_questions.json"
VAL_ANN = r"C:\Users\konab\OneDrive\Desktop\Career\projects\VQA_Project\data\vqa\v2_mscoco_val2014_annotations.json"

dataset = VQADataset(
    image_root=IMAGE_ROOT,
    question_path=VAL_Q,
    annotation_path=VAL_ANN,
    embedding_root=EMBED_ROOT,
    limit=10
)

sample = dataset[0]

print("Embedding shape:", sample["image"].shape)
