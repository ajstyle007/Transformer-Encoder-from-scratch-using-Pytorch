# Transformer Encoder from scratch

**From-scratch PyTorch Transformer encoder** implementation (based on *Attention is All You Need*) with a Streamlit demo showcasing:


<img width="1681" height="770" alt="Screenshot 2025-11-22 153022" src="https://github.com/user-attachments/assets/a7ad6d1f-399a-46ec-b4bd-ed2feb4056e9" />


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

<!-- <img width="1509" height="989" alt="Screenshot 2025-11-22 153147" src="https://github.com/user-attachments/assets/28616f56-9c6f-4d43-bd7a-d7195d8c9af4" />

<img width="1488" height="1015" alt="Screenshot 2025-11-22 153339" src="https://github.com/user-attachments/assets/28330169-a2b1-4ce7-993e-1b0197e1defa" /> -->

<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/28616f56-9c6f-4d43-bd7a-d7195d8c9af4" width="400">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/66471f1e-095d-4b79-90cb-f5112b9305c3" width="400">
    </td>
  </tr>
</table>

<table>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/28330169-a2b1-4ce7-993e-1b0197e1defa" width="400">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/88f7088e-f1c0-4475-8563-eb24fead6b82" width="400">
    </td>
  </tr>
</table>

<!-- <img width="802" height="457" alt="Screenshot 2025-11-22 153217" src="https://github.com/user-attachments/assets/66471f1e-095d-4b79-90cb-f5112b9305c3" />

<img width="1488" height="1015" alt="Screenshot 2025-11-22 153339" src="https://github.com/user-attachments/assets/88f7088e-f1c0-4475-8563-eb24fead6b82" /> -->


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

