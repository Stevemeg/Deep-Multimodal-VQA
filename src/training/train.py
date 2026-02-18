import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import json
import os

from src.models.vqa_model import VQAModel
from src.vqa_dataset import VQADataset
from src.build_vocab import build_answer_vocab


# ======================================================
# METRIC FUNCTIONS
# ======================================================

def compute_top1_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    mask = labels != -1

    if mask.sum() == 0:
        return 0.0

    correct = (preds[mask] == labels[mask]).sum().item()
    total = mask.sum().item()

    return correct / total


def compute_topk_accuracy(logits, labels, k=5):
    topk = torch.topk(logits, k, dim=1).indices
    mask = labels != -1

    if mask.sum() == 0:
        return 0.0

    correct = 0
    total = mask.sum().item()

    for i in range(len(labels)):
        if mask[i] and labels[i] in topk[i]:
            correct += 1

    return correct / total


# ======================================================
# TRAINING FUNCTION
# ======================================================

def train_model(
    image_root,
    embedding_root,
    train_q,
    train_ann,
    val_q,
    val_ann,
    batch_size=16,
    epochs=3,
    lr=1e-5,
    save_dir="checkpoints"
):

    device = torch.device("cpu")

    os.makedirs(save_dir, exist_ok=True)

    print("Building vocabulary...")
    answer_vocab = build_answer_vocab(train_ann, top_k=1000)

    train_dataset = VQADataset(
        image_root=image_root,
        question_path=train_q,
        annotation_path=train_ann,
        answer_vocab=answer_vocab,
        embedding_root=embedding_root,
        limit=10000
    )

    val_dataset = VQADataset(
        image_root=image_root,
        question_path=val_q,
        annotation_path=val_ann,
        answer_vocab=answer_vocab,
        embedding_root=embedding_root,
        limit=5000
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    model = VQAModel(num_answers=len(answer_vocab)).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=0.9
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    best_val_acc = 0

    print("Starting training...")

    for epoch in range(epochs):

        # ============================
        # TRAIN
        # ============================

        model.train()
        train_loss = 0
        train_top1 = 0

        for batch in tqdm(train_loader):

            image_embeddings = batch["image"].to(device)
            questions = batch["question"]
            labels = batch["answer"].to(device)

            optimizer.zero_grad()

            logits = model(image_embeddings, questions)

            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_top1 += compute_top1_accuracy(logits, labels)

        train_loss /= len(train_loader)
        train_top1 /= len(train_loader)

        # ============================
        # VALIDATION
        # ============================

        model.eval()

        val_loss = 0
        val_top1 = 0
        val_top5 = 0

        type_correct = {"yes/no": 0, "number": 0, "other": 0}
        type_total = {"yes/no": 0, "number": 0, "other": 0}

        failures = []

        with torch.no_grad():
            for batch in val_loader:

                image_embeddings = batch["image"].to(device)
                questions = batch["question"]
                labels = batch["answer"].to(device)
                answer_types = batch["answer_type"]
                image_ids = batch["image_id"]

                logits, attention_maps = model(
                    image_embeddings,
                    questions,
                    return_attention=True
                )

                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)

                val_top1 += compute_top1_accuracy(logits, labels)
                val_top5 += compute_topk_accuracy(logits, labels, k=5)

                for i in range(len(labels)):

                    if labels[i] == -1:
                        continue

                    atype = answer_types[i]

                    if atype not in type_total:
                        type_total[atype] = 0
                        type_correct[atype] = 0

                    type_total[atype] += 1

                    if preds[i] == labels[i]:
                        type_correct[atype] += 1
                    else:
                        failures.append({
                            "image_id": int(image_ids[i]),
                            "question": questions[i],
                            "ground_truth": int(labels[i].item()),
                            "prediction": int(preds[i].item()),
                            "confidence": float(probs[i][preds[i]].item()),
                            "answer_type": atype
                        })

        val_loss /= len(val_loader)
        val_top1 /= len(val_loader)
        val_top5 /= len(val_loader)

        # ============================
        # PRINT METRICS
        # ============================

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Top1: {train_top1:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")
        print(f"Val   Top1: {val_top1:.4f}")
        print(f"Val   Top5: {val_top5:.4f}")

        print("\nPer Answer Type Accuracy:")
        for k in type_total:
            if type_total[k] > 0:
                acc = type_correct[k] / type_total[k]
                print(f"{k}: {acc:.4f}")

        # ============================
        # SAVE FAILURES
        # ============================

        with open(os.path.join(save_dir, f"failures_epoch_{epoch+1}.json"), "w") as f:
            json.dump(failures[:100], f, indent=2)

        # ============================
        # SAVE CHECKPOINT
        # ============================

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "answer_vocab": answer_vocab,
            "val_top1": val_top1
        }

        torch.save(
            checkpoint,
            os.path.join(save_dir, f"epoch_{epoch+1}.pth")
        )

        # Save best model
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            torch.save(
                checkpoint,
                os.path.join(save_dir, "best_model.pth")
            )
            print("Best model updated.")

        scheduler.step()

    print("\nTraining complete.")
