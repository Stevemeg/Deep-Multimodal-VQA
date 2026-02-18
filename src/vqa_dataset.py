import os
import json
import re
import torch
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset


def normalize_answer(answer):
    answer = answer.lower()
    answer = re.sub(r"[^\w\s]", "", answer)
    answer = answer.strip()
    return answer


class VQADataset(Dataset):
    def __init__(
        self,
        image_root,
        question_path,
        annotation_path,
        answer_vocab=None,
        embedding_root=None,
        limit=None
    ):
        super().__init__()

        self.image_root = image_root
        self.embedding_root = embedding_root
        self.answer_vocab = answer_vocab

        with open(question_path, "r") as f:
            questions_data = json.load(f)

        with open(annotation_path, "r") as f:
            annotations_data = json.load(f)

        self.questions = questions_data["questions"]
        self.annotations = annotations_data["annotations"]

        self.annotation_map = {
            ann["question_id"]: ann for ann in self.annotations
        }

        if limit:
            self.questions = self.questions[:limit]

    def __len__(self):
        return len(self.questions)

    # ----------------------------
    # Image path resolution
    # ----------------------------
    def get_image_path(self, image_id):
        image_id_str = f"{image_id:012d}"

        for split in ["train2014", "val2014", "test2014"]:
            possible_names = [
                f"{image_id_str}.jpg",
                f"COCO_{split}_{image_id_str}.jpg"
            ]

            for name in possible_names:
                path = os.path.join(self.image_root, split, name)
                if os.path.exists(path):
                    return path

        return None

    # ----------------------------
    # Embedding path resolution
    # ----------------------------
    def get_embedding_path(self, image_id):
        if self.embedding_root is None:
            return None

        image_id_str = f"{image_id:012d}.pt"

        possible_dirs = [
            "train2014_subset",
            "val2014_subset",
            "train2014",
            "val2014",
            "test2014"
        ]

        for d in possible_dirs:
            path = os.path.join(self.embedding_root, d, image_id_str)
            if os.path.exists(path):
                return path

        return None

    # ----------------------------
    # Majority answer
    # ----------------------------
    def get_majority_answer(self, answers):
        answer_list = [
            normalize_answer(a["answer"]) for a in answers
        ]
        most_common = Counter(answer_list).most_common(1)[0][0]
        return most_common

    # ----------------------------
    # Main retrieval
    # ----------------------------
    def __getitem__(self, idx):
        question_data = self.questions[idx]

        question_id = question_data["question_id"]
        image_id = question_data["image_id"]
        question_text = question_data["question"]

        # Load embedding (preferred)
        if self.embedding_root:
            embedding_path = self.get_embedding_path(image_id)
            if embedding_path is None:
                raise FileNotFoundError(
                    f"Embedding not found for image {image_id}"
                )

            image_data = torch.load(embedding_path)

        else:
            # Fallback to raw image (debug only)
            image_path = self.get_image_path(image_id)
            if image_path is None:
                raise FileNotFoundError(
                    f"Image not found for {image_id}"
                )

            image = Image.open(image_path).convert("RGB")
            image_data = image

        annotation = self.annotation_map[question_id]
        answer_text = self.get_majority_answer(annotation["answers"])
        answer_type = annotation.get("answer_type", "other")


        if self.answer_vocab:
            answer = self.answer_vocab.get(answer_text, -1)
        else:
            answer = answer_text

        return {
            "image": image_data,
            "question": question_text,
            "answer": torch.tensor(answer, dtype=torch.long),
            "answer_type": answer_type,
            "image_id": image_id
        }
