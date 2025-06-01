# 💬 Toxicity Classifier in Rust

A minimal neural network built **from scratch in Rust** to detect toxic comments. No ML libraries — just raw matrix math, forward/backward propagation, and a custom tokenizer.

---

## 🧠 Architecture

- **Input Layer**: 1000 neurons (Bag-of-Words)
- **Hidden Layer**: 16 neurons, sigmoid
- **Output**: 1 neuron, sigmoid (toxic probability)
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: Manual Stochastic Gradient Descent (learning rate: 0.1)

---

## 🧹 Text Preprocessing

- Regex-based cleaning
- Tokenization by whitespace
- Vocabulary limited to top-N (default: 1000)
- L1-normalized frequency vectors

---

## 📊 Example Predictions

```txt
"You're an idiot and I hate you!"      → 88.80% toxic
"Thanks for your help!"               → 5.11% toxic
"Go die in a hole"                    → 84.12% toxic
"Could you explain this?"            → 5.56% toxic
```

---

## 🚀 Usage

### Build

```bash
cargo build
```

### Train & Run

```bash
cargo run
```

You can change the custom sentences for testing in the `main.rs` file

---

## ⚠️ Note

This project is for educational purposes. It's a simple demo of how neural networks work under the hood — not production-grade.

---

## 📄 Dataset

[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)

---
