#  Deep Multimodal Visual Question Answering (VQA)

> **A custom multimodal transformer that fuses vision and language to answer questions about images — built from scratch using CLIP, DistilBERT, and cross-attention fusion.**

---

##  What Is This?

Visual Question Answering (VQA) is one of the hardest problems in AI — it requires a model to simultaneously *see* an image, *understand* a natural language question, and *reason* across both modalities to produce the correct answer.

This project implements a full VQA system with a custom-designed fusion architecture, structured evaluation, failure analysis, and production-style CLI inference — trained and evaluated on the VQA v2 benchmark.

---

##  Architecture

The core insight: vision and language live in different embedding spaces. The challenge is building a bridge between them that lets each modality *attend* to the other before making a prediction.

```
Image
  │
  ▼
┌──────────────────┐
│  CLIP Vision     │  → Patch-level spatial embeddings
│  Encoder (frozen)│    (preserves pre-trained visual knowledge)
└────────┬─────────┘
         │
         │                    Question
         │                       │
         │              ┌────────▼──────────┐
         │              │  DistilBERT Text  │  → Token-level embeddings
         │              │  Encoder (tuned)  │    (fine-tuned on task)
         │              └────────┬──────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│        Cross-Attention Fusion (×3)      │  → Image patches attend to question tokens
│                                         │    Question tokens attend to image patches
└───────────────────┬─────────────────────┘
                    │
                    ▼
         ┌──────────────────┐
         │  Classification  │  → 1000-class answer head
         │  Head            │    Top-1 and Top-5 predictions
         └──────────────────┘
```

**Why these choices?**
- **CLIP frozen** — CLIP's visual representations are rich and general; fine-tuning would require far more data and risk destroying learned features
- **DistilBERT fine-tuned** — language understanding needs task-specific adaptation to VQA-style questions
- **3-layer cross-attention** — multiple fusion passes let vision and language refine each other iteratively before classification

---

##  Evaluation Results (VQA v2 Subset)

| Metric | Score |
|--------|-------|
| **Top-1 Accuracy** | ~27–33% |
| **Top-5 Accuracy** | ~65% |
| Yes/No Accuracy | ~50% |
| Number Accuracy | ~22% |
| Other Accuracy | ~7% |

The per-type breakdown reveals where the model excels (binary yes/no) and where the task gets fundamentally harder (open-ended "other" answers) — a pattern consistent with published VQA literature. Failure analysis logs are included to understand *why* the model errs, not just that it does.

---

##  Attention Visualization

The model extracts **CLS-to-patch attention weights** from the cross-attention layers and overlays them on the original image as a heatmap — showing *where* the model looked when answering the question.

This makes the model's reasoning interpretable and debuggable.

```
Input Image + Question → Attention Heatmap Overlay → result.png
```

---

##  Tech Stack

| Component | Tool |
|-----------|------|
| Deep Learning | PyTorch |
| Vision Encoder | CLIP (OpenAI, via HuggingFace) |
| Text Encoder | DistilBERT (HuggingFace Transformers) |
| Attention Visualization | OpenCV |
| Data & Plotting | NumPy, Matplotlib |

---

##  Usage

**Training**
```bash
python run_train.py
# Checkpoints saved to checkpoints/
```

**Inference (CLI)**
```bash
python inference.py \
  --checkpoint checkpoints/best_model.pth \
  --embedding path_to_embedding.pt \
  --image path_to_image.jpg \
  --question "Your question here" \
  --output result.png
```
The output saves the image with the attention heatmap overlay and prints Top-1 and Top-5 predicted answers.

---

##  Project Structure

```
multimodal-vqa/
│
├── models/
│   ├── vision_encoder.py      # CLIP patch embedding extraction
│   ├── text_encoder.py        # DistilBERT token embedding extraction
│   ├── cross_attention.py     # 3-layer cross-attention fusion module
│   └── classifier.py          # 1000-class answer classification head
│
├── training/
│   └── run_train.py           # Training loop with checkpoint saving
│
├── evaluation/
│   ├── evaluate.py            # Per-type accuracy metrics
│   └── failure_analysis.py    # Logs and categorizes model errors
│
├── visualization/
│   └── attention_overlay.py   # CLS-to-patch attention heatmap
│
├── inference.py               # CLI inference tool
├── checkpoints/               # Saved model weights
└── README.md
```

---

##  Key Features

- **Custom multimodal fusion** — cross-attention designed and implemented from scratch, not a pre-built VQA head
- **Structured evaluation** — per-answer-type breakdown (yes/no, number, other), not just a single accuracy number
- **Failure analysis logging** — systematic capture of error patterns to inform future improvements
- **Attention heatmap visualization** — makes the model interpretable by showing spatial focus regions
- **Production-style CLI** — inference tool takes real images and questions, outputs annotated results

---

##  Why This Matters

This project demonstrates:
- **Multimodal AI** — combining vision and language transformers in a single learnable system
- **Transformer internals** — implementing cross-attention fusion from scratch, not just calling a library
- **Transfer learning strategy** — knowing *which* components to freeze vs. fine-tune and why
- **Evaluation depth** — going beyond overall accuracy to per-category breakdowns and failure analysis
- **Interpretability** — building attention visualization to make model behavior explainable
- **Research-to-engineering** — taking an academic task (VQA v2 benchmark) to a working CLI tool

---

##  Future Roadmap

- [ ] Upgrade to larger vision-language backbone (e.g., BLIP-2, LLaVA)
- [ ] Beam search or constrained decoding for open-ended answers
- [ ] Web interface for image upload + live question answering
- [ ] Contrastive pre-training on medical image-question pairs

---

