# Transformer Encoder from scratch

**From-scratch PyTorch Transformer encoder** implementation (based on *Attention is All You Need*) with a Streamlit demo showcasing:

- ✅ **Extract Keywords** — keyphrase extraction using BIO-trained encoder
- ✅ **Attention Heatmap Viewer** — visualize self-attention per head and layer
- ✅ **NER Demo** — token-level BIO predictions (used for highlighting)

This repository implements Multi-Head Attention, Positional Encoding, Feed-Forward networks, Residual+LayerNorm, and an encoder stack from first principles. The encoder was trained on KP20k (BIO labels) to validate the implementation and to experiment with attention behavior, head effects and extraction heuristics.

---

## Features
- Pure PyTorch implementation of encoder building blocks (multihead attention, PE, FFN).
- Training pipeline (KP20k) for token-level BIO supervision (keyword extraction).
- Streamlit front-end with three utilities:
  - Keyphrase extraction
  - Attention heatmap visualization
  - Token highlighter (BIO predictions)
- Inference-time heuristics (no retraining required) to clean and merge subwords, POS filtering, and strict BIO decoding.

---

## How it was built / notes

- Implemented attention, positional encodings and FFN from scratch in PyTorch.
- Used BERT tokenizer (pretrained) for consistent tokenization with KP20k.
- Trained single-layer (or few-layer) encoder on KP20k for BIO labeling to validate implementation and experiment with attention.
- Front-end demonstrates inspection of attention patterns and practical use-cases without needing heavy retraining.

---

## What I learned

- Exact tensor shapes and permutations required by multi-head attention.
- Why positional encodings are needed for order information.
- How attention heads behave differently and how that affects extraction.
- Practical tricks for deployment: subword merging, strict BIO decoding, POS filtering and inference-time smoothing.

---

