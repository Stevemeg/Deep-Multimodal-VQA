import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from src.models.vqa_model import VQAModel

DEVICE = torch.device("cpu")


# ======================================================
# LOAD MODEL
# ======================================================

def load_model(checkpoint_path):

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"{checkpoint_path} not found")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    answer_vocab = checkpoint["answer_vocab"]
    idx_to_answer = {v: k for k, v in answer_vocab.items()}

    model = VQAModel(num_answers=len(answer_vocab))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, idx_to_answer


# ======================================================
# PREDICTION
# ======================================================

def predict(model, idx_to_answer, embedding_path, question):

    image_embedding = torch.load(embedding_path).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits, attention_maps = model(
            image_embedding,
            [question],
            return_attention=True
        )

        probs = torch.softmax(logits, dim=1)
        top5 = torch.topk(probs, 5, dim=1)

        predictions = []
        for i in range(5):
            idx = top5.indices[0][i].item()
            score = top5.values[0][i].item()
            predictions.append((idx_to_answer[idx], score))

    return predictions, attention_maps


# ======================================================
# ATTENTION OVERLAY
# ======================================================

def save_attention_overlay(attention_maps, image_path, output_path):

    last_layer = attention_maps[-1]

    if last_layer.dim() == 4:
        cls_attention = last_layer[:, :, 0, :].mean(dim=1)
    else:
        cls_attention = last_layer[:, 0, :]

    cls_attention = cls_attention.squeeze(0).cpu().numpy()

    # Remove CLS token if present
    if len(cls_attention) == 50:
        cls_attention = cls_attention[1:]

    grid_size = int(np.sqrt(len(cls_attention)))

    if grid_size * grid_size != len(cls_attention):
        print("Cannot reshape attention into square grid.")
        return

    attention_map = cls_attention.reshape(grid_size, grid_size)

    # Normalize attention
    attention_map = (attention_map - attention_map.min()) / (
        attention_map.max() - attention_map.min() + 1e-8
    )

    # Load original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape

    # Resize attention to image size
    attention_resized = cv2.resize(attention_map, (w, h))
    attention_resized = np.uint8(255 * attention_resized)

    heatmap = cv2.applyColorMap(attention_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Overlay saved to {output_path}")


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VQA Inference CLI")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--embedding", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--output", type=str, default="attention_overlay.png")

    args = parser.parse_args()

    model, idx_to_answer = load_model(args.checkpoint)

    predictions, attention_maps = predict(
        model,
        idx_to_answer,
        args.embedding,
        args.question
    )

    print("\nTop 5 Predictions:")
    for ans, conf in predictions:
        print(f"{ans:20s} {conf:.4f}")

    save_attention_overlay(
        attention_maps,
        args.image,
        args.output
    )
