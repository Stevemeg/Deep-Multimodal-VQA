 Deep Multimodal VQA System
markdown# Deep Multimodal Visual Question Answering (VQA) System

> Fusing vision and language through custom cross-attention transformers for open-ended visual reasoning.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview
A multimodal transformer that answers natural language questions about images 
by jointly reasoning over CLIP vision embeddings and DistilBERT text embeddings 
through a custom 3-layer cross-attention mechanism.

## Architecture
Image → CLIP Encoder → Vision Embeddings ─┐
├→ Cross-Attention (3 layers) → Answer Head
Text  → DistilBERT   → Text Embeddings  ──┘

## Results (VQA v2 Benchmark Subset)
| Answer Type | Top-1 Accuracy | Top-5 Accuracy |
|-------------|---------------|----------------|
| Yes / No    | ~33%          | ~65%           |
| Number      | ~33%          | ~65%           |
| Open-ended  | ~33%          | ~65%           |

## Key Features
- Custom 3-layer cross-attention mechanism for joint visual-language reasoning
- Attention-based interpretability module with spatial heatmap overlays
- Failure analysis logs to identify and document model blind spots
- End-to-end training pipeline with checkpointing
- CLI inference for easy testing

## Project Structure
vqa-system/
├── models/
│   ├── vision_encoder.py      # CLIP-based vision encoder
│   ├── text_encoder.py        # DistilBERT text encoder
│   └── cross_attention.py     # Custom 3-layer cross-attention
├── train.py                   # Training pipeline with checkpointing
├── inference.py               # CLI inference script
├── evaluate.py                # VQA v2 evaluation metrics
├── interpretability/
│   └── heatmap.py             # Spatial attention heatmaps
└── requirements.txt

## Setup & Usage
```bash
git clone https://github.com/yourusername/multimodal-vqa
cd multimodal-vqa
pip install -r requirements.txt

# Train
python train.py --epochs 20 --batch_size 32

# Inference
python inference.py --image sample.jpg --question "What is in the image?"
```

## Tech Stack
`PyTorch` `HuggingFace Transformers` `CLIP` `DistilBERT` `Python`
