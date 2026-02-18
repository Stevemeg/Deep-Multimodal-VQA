import sys
import os

# Force src to be first in path
sys.path.insert(0, os.path.abspath("src"))

from training.train import train_model


IMAGE_ROOT = r"C:\Users\konab\OneDrive\Desktop\Career\projects\VQA_Project\data\images"
EMBED_ROOT = r"C:\Users\konab\OneDrive\Desktop\Career\projects\VQA_Project\data\embeddings"

TRAIN_Q = r"C:\Users\konab\OneDrive\Desktop\Career\projects\VQA_Project\data\vqa\v2_OpenEnded_mscoco_train2014_questions.json"
TRAIN_ANN = r"C:\Users\konab\OneDrive\Desktop\Career\projects\VQA_Project\data\vqa\v2_mscoco_train2014_annotations.json"

VAL_Q = r"C:\Users\konab\OneDrive\Desktop\Career\projects\VQA_Project\data\vqa\v2_OpenEnded_mscoco_val2014_questions.json"
VAL_ANN = r"C:\Users\konab\OneDrive\Desktop\Career\projects\VQA_Project\data\vqa\v2_mscoco_val2014_annotations.json"

train_model(
    IMAGE_ROOT,
    EMBED_ROOT,
    TRAIN_Q,
    TRAIN_ANN,
    VAL_Q,
    VAL_ANN,
    batch_size=16,
    epochs=3
)
