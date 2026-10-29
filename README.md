---
language:
- en
license: apache-2.0
tags:
- classifier
- onnx
- difficulty-classifier
- structzero
- ysolveth
- ml
- routing
pipeline_tag: text-classification
---

# Ysolveth (Candidate 1.3.0)

[![Platform](https://img.shields.io/badge/Platform-StructZero-blue)](https://www.structzero.app/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![ONNX Runtime](https://img.shields.io/badge/Inference-ONNX_Runtime-brightgreen)](https://onnxruntime.ai/)

> Developed for [StructZero — AI Software Engineering Platform](https://www.structzero.app/).

**Ysolveth** is an ultra-fast, local prompt difficulty estimation and triage model designed to analyze software engineering requests and complex development tasks before routing them to generative AI pipelines. It classifies prompts into three operational complexity tiers: **`trivial`**, **`moderate`**, and **`hard`**.

---

## What It Is (Architecture)

**Ysolveth is not an LLM** or autoregressive text-generation model. It is a compact, sub-millisecond classifier comprising:
1. **Embedding Backbone**: Frozen `sentence-transformers/all-MiniLM-L6-v2` encoder quantized to **int8 ONNX** (`encoder_int8.onnx`, ~22.8 MB).
2. **Classification Head**: Lightweight calibrated linear layer (`head_config.json`, 384 input dimensions &rarr; 3 output classes with temperature scaling & softmax).
3. **Inference Latency**: ~2 to 8ms on standard CPU (zero GPU required).

---

## Use Cases

### 1. Intelligent Model & Workflow Routing
Directing all user prompts to heavyweight frontier models (e.g. Claude 3.5 Sonnet, GPT-4o) incurs high costs and latency. Ysolveth enables tiered triage:
- **`trivial`**: Route to lightweight, inexpensive models or deterministic templates (e.g., regex fixes, simple refactors, documentation typo updates).
- **`moderate`**: Route to standard reasoning coding models.
- **`hard`**: Route into multi-model consensus, multi-turn architecture debate, and strict sandbox verification harnesses where critical multi-component verification is required.

### 2. Guardrails Against Single-Model Blindspots
Multi-tenant database schema migrations, Stripe webhook reconciliation, and OAuth/SAML integrations require multiple validation steps. When flagged by Ysolveth as `hard`, the orchestrator ensures that code changes are guarded by multi-agent consensus rather than letting a single model blindly apply changes.

### 3. Cost & Latency Optimization
By avoiding unnecessary calls to frontier models for trivial and moderate prompts, AI coding platforms can dramatically lower monthly inference spend while maintaining instant sub-10ms routing decisions.

---

## Benchmark

Ysolveth Candidate 1.3.0 was evaluated on an independent, strictly held-out test set of **40 hand-labelled prompts** (10 trivial, 15 moderate, 15 hard) with zero overlap with the training pool.

### 4.1 Primary Metric: Critical Failures
A **critical failure** is defined as classifying a **`hard`** request as **`trivial`** or **`moderate`**, which routes complex architectures into unguided single-model lanes without verification.

| Model | Critical Failures (Hard &rarr; Trivial/Mod) | Critical Failure Rate | Overall Test Accuracy (40 Prompts) |
| :--- | :---: | :---: | :---: |
| **Live Mini Model (Baseline)** | **14 / 15** | **93.3%** | **65.0%** (26/40) |
| **Ysolveth (Candidate 1.3.0)** | **0 / 15** | **0.0%** | **100.0%** (40/40) |

- **Baseline Live Mini Model**: Erroneously classified 14 of 15 hard prompts as `moderate`.
- **Ysolveth Candidate 1.3.0**: Correctly classified all 15 hard prompts as `hard` (0 of 15 critical failures, 100.0% accuracy across all 40 prompts).

### 4.2 Classification Report (Ysolveth Candidate 1.3.0)

```text
              precision    recall  f1-score   support

     trivial     1.0000    1.0000    1.0000        10
    moderate     1.0000    1.0000    1.0000        15
        hard     1.0000    1.0000    1.0000        15

    accuracy                         1.0000        40
   macro avg     1.0000    1.0000    1.0000        40
weighted avg     1.0000    1.0000    1.0000        40
```

### 4.3 Confusion Matrix Comparison

#### Baseline Mini Model Confusion Matrix:
```text
             trivial  moderate  hard
trivial         10         0     0
moderate         0        15     0
hard             0        14     1   <-- 14 critical failures
```

#### Ysolveth Candidate 1.3.0 Confusion Matrix:
```text
             trivial  moderate  hard
trivial         10         0     0
moderate         0        15     0
hard             0         0    15   <-- 0 critical failures
```

---

## Limits and Caveats

1. **Test Set Size**: The held-out benchmark test set is small (**40 prompts**).
2. **Label Methodology**: Ground-truth labels are derived from owner judgment combined with a 4-model majority consensus vote.
3. **Statistical Bounds**: Results reflect a single benchmark run; no bootstrap confidence intervals are computed.
4. **Generalization**: This is an internal domain-specific routing benchmark for software engineering prompts, not a general natural language processing benchmark.
5. **Confidentiality**: Raw training prompts, hard-case examples, and customer data are withheld to preserve internal confidentiality and security.

---

## Candidate Checksums

Every file in the repository matches its pinned cryptographic SHA-256 hash:

| File | SHA-256 Checksum |
|---|---|
| `encoder_int8.onnx` | `9d6913869f2523a41c8e24252e69061ed31b69a8d04a7ec6359d2d1ff8e308f6` |
| `head_config.json` | `d5c13b57ed81a167897030fe04093da4b7d21d5a6a6c3591c7ade412caaee551` |
| `tokenizer.json` | `e619bb7888bee125bfd361ad7eb9f8e3de9b5ece0ed0e5949db658630b9c60da` |
| `tokenizer_config.json` | `872b6936be955bc3aea75ed599264d865626d68feede7e58b01e378e6332bd74` |
| `vocab.txt` | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` |
| `special_tokens_map.json` | `303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3` |

---

## How to Run Inference

### Python (ONNX Runtime + Tokenizer)

```python
import json
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# 1. Load tokenizer and ONNX encoder
tokenizer = AutoTokenizer.from_pretrained("vishalvr28/Ysolveth")
session = ort.InferenceSession("encoder_int8.onnx")

# 2. Load head weights and configuration
with open("head_config.json", "r") as f:
    head_config = json.load(f)

weights = np.array(head_config["weights"])      # shape: (3, 384)
bias = np.array(head_config["bias"])            # shape: (3,)
classes = head_config.get("classes", ["trivial", "moderate", "hard"])
temperature = head_config.get("temperature", 1.0)

# 3. Preprocess prompt
prompt = "Build a multi-tenant SaaS with user login, Stripe webhooks, and PostgreSQL migrations."
inputs = tokenizer(prompt, return_tensors="np", padding=True, truncation=True, max_length=256)

# 4. Generate MiniLM embeddings (Mean Pooling)
onnx_inputs = {
    "input_ids": inputs["input_ids"].astype(np.int64),
    "attention_mask": inputs["attention_mask"].astype(np.int64),
    "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"])).astype(np.int64)
}
outputs = session.run(None, onnx_inputs)
token_embeddings = outputs[0]  # shape: (batch_size, seq_len, 384)
attention_mask = inputs["attention_mask"][..., np.newaxis]
pooled_embedding = np.sum(token_embeddings * attention_mask, axis=1) / np.maximum(attention_mask.sum(axis=1), 1e-9)

# Normalize embedding
embedding = pooled_embedding / np.linalg.norm(pooled_embedding, axis=1, keepdims=True)

# 5. Compute Linear Head Logits & Probabilities
logits = (np.dot(embedding, weights.T) + bias) / temperature
exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

predicted_class = classes[np.argmax(probs)]
confidence = np.max(probs)
print(f"Prediction: {predicted_class} (Confidence: {confidence:.2%})")
```

---

## How to Train & Calibrate on Your Own Data

You do **not** need to fine-tune the heavy MiniLM transformer weights. Fine-tuning only the lightweight linear classification head preserves rapid training (seconds) and eliminates overfitting.

### Step-by-Step Training Guide

#### 1. Prepare Your Dataset
Prepare a JSON or CSV file of software engineering tasks with assigned difficulty levels:
```json
[
  {"prompt": "Fix typo in README footer", "label": "trivial"},
  {"prompt": "Add JWT expiration check middleware", "label": "moderate"},
  {"prompt": "Implement distributed Saga pattern across order and inventory microservices", "label": "hard"}
]
```

#### 2. Extract Embeddings with Frozen MiniLM
Use `sentence-transformers/all-MiniLM-L6-v2` to precompute the 384-dimensional normalized vector for each prompt once:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
X = model.encode([item['prompt'] for item in dataset], normalize_embeddings=True)
y = np.array([label_to_id[item['label']] for item in dataset])
```

#### 3. Train a Calibrated Linear Classifier
Fit a Logistic Regression model or train a small Linear Head using Cross-Entropy loss:
```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    C=1.0,
    class_weight='balanced',  # Crucial to protect underrepresented hard cases
    max_iter=1000,
    multi_class='multinomial'
)
clf.fit(X, y)

# Extract weights and biases
weights = clf.coef_.tolist()       # List of 3 lists, each 384 floats
bias = clf.intercept_.tolist()      # List of 3 floats
```

#### 4. Calibrate with Temperature Scaling & Thresholding
To enforce a zero-critical-failure safety policy, apply an epsilon threshold or temperature scaling:
- If `P(hard) >= 0.35`, bias classification towards `hard` to ensure complex multi-step tasks are never mistakenly labeled `trivial` or `moderate`.

#### 5. Save to `head_config.json`
Export the configuration into the standard StructZero runtime schema:
```json
{
  "version": "1.3.0",
  "classes": ["trivial", "moderate", "hard"],
  "input_dim": 384,
  "weights": [...],
  "bias": [...],
  "temperature": 1.0,
  "thresholds": {
    "hard_min_probability": 0.35
  }
}
```

---

## Attribution & License

- **Platform**: [StructZero — AI Software Engineering Platform](https://www.structzero.app/)
- **Base Model**: `sentence-transformers/all-MiniLM-L6-v2` by Sentence-Transformers, licensed under Apache 2.0.
