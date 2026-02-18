
import sys
import os
sys.path.append(os.path.abspath("src"))

from vqa_dataset import VQADataset
from build_vocab import build_answer_vocab
from torch.utils.data import DataLoader


def get_dataloaders(
    image_root,
    train_q,
    train_ann,
    val_q,
    val_ann,
    batch_size=8
):

    print("Building answer vocabulary...")
    answer_vocab = build_answer_vocab(train_ann, top_k=1000)
    print("Vocabulary size:", len(answer_vocab))

    train_dataset = VQADataset(
        image_root=image_root,
        question_path=train_q,
        annotation_path=train_ann,
        answer_vocab=answer_vocab
    )

    val_dataset = VQADataset(
        image_root=image_root,
        question_path=val_q,
        annotation_path=val_ann,
        answer_vocab=answer_vocab
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows safe
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, answer_vocab
