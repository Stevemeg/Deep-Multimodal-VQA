import os
import torch
import json
from tqdm import tqdm
from PIL import Image

from models.vision_encoder import CLIPVisionEncoder


IMAGE_ROOT = r"C:\Users\konab\OneDrive\Desktop\Career\projects\VQA_Project\data\images"
SAVE_ROOT = r"C:\Users\konab\OneDrive\Desktop\Career\projects\VQA_Project\data\embeddings"
VAL_Q_PATH = r"C:\Users\konab\OneDrive\Desktop\Career\projects\VQA_Project\data\vqa\v2_OpenEnded_mscoco_val2014_questions.json"

SUBSET_SIZE = 5000  # adjust if needed


def get_image_path(image_id):
    image_id_str = f"{image_id:012d}"

    for split in ["train2014", "val2014", "test2014"]:
        possible_names = [
            f"{image_id_str}.jpg",
            f"COCO_{split}_{image_id_str}.jpg"
        ]

        for name in possible_names:
            path = os.path.join(IMAGE_ROOT, split, name)
            if os.path.exists(path):
                return path

    return None


def main():
    print("Loading validation questions...")

    with open(VAL_Q_PATH, "r") as f:
        data = json.load(f)

    questions = data["questions"][:SUBSET_SIZE]

    image_ids = list(set([q["image_id"] for q in questions]))

    print(f"Unique images in subset: {len(image_ids)}")

    vision = CLIPVisionEncoder(device="cpu")

    save_split_path = os.path.join(SAVE_ROOT, "val2014_subset")
    os.makedirs(save_split_path, exist_ok=True)

    for image_id in tqdm(image_ids):
        image_path = get_image_path(image_id)

        if image_path is None:
            print(f"Skipping missing image {image_id}")
            continue

        save_path = os.path.join(
            save_split_path,
            f"{image_id:012d}.pt"
        )

        if os.path.exists(save_path):
            continue

        image = Image.open(image_path).convert("RGB")
        embeddings = vision.encode(image)

        torch.save(embeddings.cpu(), save_path)


if __name__ == "__main__":
    main()
