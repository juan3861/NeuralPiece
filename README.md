# 🧠 AetherMind Tokenizer Lab: Foundations to 2077 Neural Architecture

Welcome to the **AetherMind Tokenizer Lab**, an experimental environment and deep-dive technical guide designed to demystify LLM tokenization—from its absolute basics to the cutting-edge concepts of "token-free" neural chunking architectures.

## 📖 The Architecture of Reading (How LLMs read text)

### 1. The Core Problem: Why Tokenize?
Machine Learning models do not understand text; they understand arrays of numbers (tensors). Tokenization is the translation layer between human language and machine computation. It maps subword units into unique numeric IDs. The tokenizer dictates the **alphabet** of the model. If a tokenizer handles a language poorly, the model will suffer from smaller effective context windows and worse arithmetic capabilities.

### 2. Evolution of Tokenization

| Method | Approach | Characteristics | Used By |
|--------|----------|-----------------|---------|
| **BPE** (Byte Pair Encoding) | Bottom-up | Merges the most frequent byte pairs iteratively. Deterministic. | GPT-3, GPT-4, RoBERTa |
| **WordPiece** | Bottom-up | Merges pairs that maximize language model likelihood. | BERT |
| **Unigram** | Top-down | Starts with a massive vocabulary and prunes the least useful tokens. Probabilistic. | LLaMA, T5, Gemma |

### 3. The SentencePiece Framework
SentencePiece is not an algorithm itself, but a powerful framework that treats text as a raw stream of Unicode characters (or bytes), completely independent of spaces or language rules.
*   **Subword Regularization:** Because Unigram is probabilistic, the same word can be tokenized differently across training epochs, acting as free data augmentation and making the model more robust.

### 4. Neural Chunker 2077: The Post-Tokenizer Era
Traditional tokenizers use fixed vocabularies and deterministic (or semi-deterministic) mathematical rules. The future points toward **Adaptive Neural Tokenization** (inspired by research like ByT5, MegaByte, and CANINE).
We implemented a **Neural Chunker Prototype** that uses a lightweight 1D-CNN over raw UTF-8 bytes to *learn* where to segment text, rather than relying on fixed vocabularies.

*   **Context-dependent Tokenization:** In theory, it can segment the same characters differently depending on surrounding context.
*   **Infinite Vocabulary:** No out-of-vocabulary (OOV) issues.
*   **Zero-Retraining Evolution:** Learns new boundaries continuously.

---

## 🧪 Laboratory Experiments & Reproduction

This repository contains two main experiments you can run locally:

### Prerequisites
Install dependencies using:
```bash
pip install -r requirements.txt
```

### Experiment 01: BPE vs Unigram Engine
Trains both BPE and Unigram tokenizers on a local Portuguese/Mixed corpus and compares their compactness and fertility imbalance.

**Run the comparison:**
```bash
python 01_train_and_compare.py
```
*Results typically show Unigram outperforming BPE in compactness and demonstrating powerful Subword Regularization.*

### Experiment 02: Neural Chunker 2077
Trains a 41k-parameter CNN to mimic SentencePiece boundaries entirely at the byte level, demonstrating the concept of a learned tokenizer.

**Run the neural chunker:**
```bash
python 02_neural_chunker_2077.py
```

---

## 📊 Key Findings from the Lab

1.  **Unigram beats BPE in Compactness:** Unigram efficiently captured compound semantics (e.g., "programador" as 1 token) where BPE failed (3 tokens).
2.  **Fertility Imbalance is Real:** Our Portuguese-trained tokenizer was highly efficient for Portuguese (0.45 tokens/char) but penalized unseen languages severely (Japanese at 3.07 tokens/char).
3.  **Neural Chunking is Viable:** Our 41K parameter CNN reached 98.7% boundary prediction accuracy and sometimes achieved higher compression than the fixed SentencePiece model by learning superior dynamic boundaries.

---
*Developed by AetherMind & Ruan.*
