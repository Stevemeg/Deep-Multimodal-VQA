Deep Multimodal Visual Question Answering (VQA)
 Overview

This project implements a deep multimodal transformer for Visual Question Answering using:

CLIP Vision Encoder (frozen)

DistilBERT Text Encoder (fine-tuned)

3-layer Cross-Attention Fusion

1000-class Answer Classification Head

Attention-based spatial visualization overlay

The system includes training, structured evaluation, failure analysis, and production-style CLI inference.

 Architecture

Pipeline:

Image → CLIP → Patch Embeddings

Question → DistilBERT → Token Embeddings

Cross-Attention Fusion (×3 layers)

Classification Head → Top-1 & Top-5 Answers

 Evaluation Results (VQA v2 Subset)
Metric	Value
Top-1 Accuracy	~27–33%
Top-5 Accuracy	~65%
Yes/No Accuracy	~50%
Number Accuracy	~22%
Other Accuracy	~7%

Includes per-answer-type evaluation and failure analysis logs.

 Attention Visualization

The model extracts CLS-to-image patch attention and overlays it on the original image.

Example:

 Training
python run_train.py


Checkpoints saved in:

checkpoints/

 Inference
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --embedding path_to_embedding.pt \
  --image path_to_image.jpg \
  --question "Your question here" \
  --output result.png

 Tech Stack

PyTorch

HuggingFace Transformers

OpenCV

NumPy

Matplotlib

 Key Features

Custom multimodal fusion transformer

Structured evaluation pipeline

Failure analysis logging

Attention heatmap overlay

CLI-based inference tool