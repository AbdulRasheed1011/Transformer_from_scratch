# Transformer Summarization from Scratch 🚀

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A complete implementation of the **Transformer** architecture (from the paper *"Attention Is All You Need"*) built from scratch using **TensorFlow**. This project focuses on **Abstractive Text Summarization** using the **CNN/DailyMail** dataset.

## 📖 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Google Colab Experiments](#-google-colab-experiments-recommended)
- [Model Architecture](#-model-architecture)
- [Experiments Log](#-experiments-log)
- [References](#-references)

---

## 🔍 Overview

This repository demonstrates how to build a Transformer model without relying on high-level Keras layers like `MultiHeadAttention` or `TransformerEncoder`. Every component—from **Positional Encoding** to **Scaled Dot-Product Attention**—is implemented manually to provide a deep understanding of the architecture.

The goal is to train a model capable of generating concise summaries from news articles, targeting a **70% accuracy** benchmark on the test set.

## ✨ Key Features

- **From-Scratch Implementation**: Core Transformer components (Encoder, Decoder, Multi-Head Attention) built with raw TensorFlow ops.
- **Custom Tokenizer**: Byte-Pair Encoding (BPE) trained specifically for this dataset.
- **MLflow Integration**: Full experiment tracking (Loss, Accuracy, ROUGE scores).
- **Optimized Pipeline**: efficiently handles the large CNN/DailyMail dataset using `tf.data`.
- **Inference Engine**: Greedy decoding for generating summaries from new text.

## 📂 Project Structure

```bash
Transformer_from_Scratch/
├── config/
│   └── config.yaml          # centralized configuration (Model, Data, Training)
├── notebooks/
│   └── Transformer_Training.ipynb  # Interactive Experiments Lab (Colab ready)
├── src/
│   ├── data_ingestion/      # Data download & Preprocessing pipeline
│   ├── model/               # Core Transformer modules (Layers, Attention, PE)
│   ├── training/            # Custom training loop with Checkpointing & MLflow
│   └── inference/           # Inference logic for generating summaries
├── requirements.txt         # Project dependencies
└── Readme.md                # Documentation
```

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/AbdulRasheed1011/Transformer_from_scratch.git
   cd Transformer_from_scratch
   ```

2. **Install Dependencies**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Start

### 1. Data Preparation
Download the dataset and train the BPE tokenizer.
```bash
python -m src.data_ingestion.preprocessing
```

### 2. Training
Start the training loop (check `config/config.yaml` to adjust epochs/batch size).
```bash
python -m src.training.train
```

### 3. Inference
Generate a summary for a sample text using the trained model.
```bash
python src/inference/inference.py
```

---

## 🏗 Model Architecture

The model adheres strictly to the original Transformer design:

*   **Embeddings**: Learned vector representations scaled by $\sqrt{d_{model}}$.
*   **Positional Encoding**: Sinusoidal functions added to embeddings to retain sequence order.
*   **Encoder**: stack of $N$ layers, each containing:
    *   Multi-Head Self-Attention
    *   Position-wise Feed-Forward Network
*   **Decoder**: stack of $N$ layers, each containing:
    *   Masked Multi-Head Self-Attention
    *   Multi-Head Cross-Attention (query from decoder, keys/values from encoder)
    *   Position-wise Feed-Forward Network

## 📊 Experiments Log

**Goal:** Reach **70% Test Accuracy**.

| Experiment | Parameters (Layers, d_model, dff) | Epochs | Batch Size | Train Acc | Val Acc | Test Acc | Time | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 4, 128, 512 | 1 | 16 | 0.0000 | 0.0000 | 0.0000 | 51.89s | Debug Run (Underfit) |
| **Exp 1 (Large)** | 6, 256, 1024 | 1 | 16 | 0.0000 | 0.0000 | 0.0000 | 51.89s | Debug Run (Benchmark) |
| **Exp 2 (Small)** | 2, 64, 256 | 1 | 16 | 0.0000 | 0.0000 | 0.0000 | 12.65s | 4x Faster than Exp 1 |
| **Exp 3 (Heavy)** | 6, 256, 1024 | 10 | 8 | - | - | - | >10m/epoch | Estimated (Slow on Local) |

*(Update this table with your results from the Colab notebook)*

## 📚 References

*   [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
*   [CNN/DailyMail Dataset](https://huggingface.co/datasets/cnn_dailymail)
