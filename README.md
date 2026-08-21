# 🦁 The Roar of the Lion: Replicating the Evolved Sign Momentum Optimizer

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains a modular, from-scratch PyTorch replication and performance verification of **Lion (EvoLved Sign Momentum)**, a first-order optimization algorithm discovered by researchers at Google and UCLA (Chen et al., 2023) using automated symbolic program search. 

---

## 📖 Abstract

We present a systematic replication and empirical verification of the **Lion** optimizer, evaluating its training dynamics, memory footprints, and optimization trajectories on high-dimensional deep learning tasks. Discovered via evolutionary program search, Lion is mathematically simpler and more memory-efficient than industry-standard adaptive optimizers like **AdamW** and **Adafactor**, as it maintains only momentum history (omitting the second-moment variance vector $v$). Unlike standard adaptive methods, Lion computes parameter updates through an element-wise **`sign` operation**. This produces uniform update magnitudes that act as a regularizer, steering training toward flatter, more generalizable regions of the loss landscape.

Over a structured timeline, we build a custom PyTorch optimizer class implementing the evolved Lion update formula. Using standard benchmarks such as image classification on **CIFAR-100**, we evaluate several core performance claims reported in the original paper:
1. **Hyperparameter Scaling Rules**: Replicating the unique hyperparameter coupling of Lion, showing that it requires a **3–10x smaller learning rate** ($\eta$) and a **3–10x larger weight decay** ($\lambda$) than AdamW to maintain equivalent effective weight decay strength ($\eta \cdot \lambda$).
2. **Batch Size Sensitivity**: Verifying the paper's finding that Lion's performance gains over AdamW scale up with larger training batch sizes, particularly demonstrating robust convergence at a batch size of 4,096.
3. **Loss Geometry and Generalization**: Implementing random Gaussian noise perturbations (weight-space flatness audits) to verify that Lion forces models to converge to flatter loss landscape minima compared to AdamW.
4. **Computational and Memory Savings**: Measuring step-wise execution runtime to replicate the reported **2–15% step speedup** and significantly reduced memory usage across training steps.

---

## 🧮 Mathematical Comparison

### 1. AdamW Update Mechanics
For a loss function $f(\theta)$ with learning rate $\eta_t$, weight decay $\lambda$, and gradient $g_t$:
$$m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t \leftarrow \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t \leftarrow \frac{v_t}{1 - \beta_2^t}$$
$$\theta_t \leftarrow \theta_{t-1} - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

### 2. Lion (EvoLved Sign Momentum) Update Mechanics
Lion removes the tracking of the second-moment $v$, simplifying the calculation to:
$$c_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$\theta_t \leftarrow \theta_{t-1} - \eta_t \left( \text{sign}(c_t) + \lambda \theta_{t-1} \right)$$
$$m_t \leftarrow \beta_2 m_{t-1} + (1 - \beta_2) g_t$$

*Here, the default hyperparameters discovered during program search are $\beta_1 = 0.9$ and $\beta_2 = 0.99$.*

---

## 🎯 Replication Targets & Metrics

| Metric | Target (Lion) | Compared To (AdamW) | Source Grounding |
| :--- | :--- | :--- | :--- |
| **Peak LR ($\eta$)** | **3–10x smaller** ($1\times10^{-4}$ to $3\times10^{-4}$) | Standard ($1\times10^{-3}$) | Section 5 |
| **Weight Decay ($\lambda$)** | **3–10x larger** ($1.0$ to $10.0$) | Standard ($0.1$ to $1.0$) | Section 5 |
| **Step-Wise Speedup** | **2–15% faster execution** | Baseline runtime (100%) | Section 3.2 |
| **Momentum Memory** | **~50% tracking savings** (only $m_t$ cached) | Tracks both $m_t$ and $v_t$ | Section 1 & 3.2 |
| **Batch Size Scale** | Widens accuracy gap at **batch size = 4,096** | Performs best at batch size = 256 | Section 4.6 |
| **Loss Flatness ($L^N_{\text{train}}$)** | **Lower perturbed loss** ($1.37$ on ViT-B/16) | Higher perturbed loss ($3.74$) | Section G & Table 10 |

---

## 🗓️ 3-Week Project Roadmap

### **Week 1: Custom Math & Core Pipeline**
- [ ] Implement `class Lion(torch.optim.Optimizer)` in PyTorch, implementing the exact dual-interpolation and sign logic.
- [ ] Build a standard image classification training pipeline on **CIFAR-100** using ResNet-18.
- [ ] Establish a rigorous **AdamW** control baseline, tracking train/val losses, and step runtimes.

### **Week 2: Hyperparameter Sweeps & Scale Sensitivity**
- [ ] Conduct logarithmic grid sweeps on learning rate ($\eta \in [10^{-5}, 10^{-3}]$) and weight decay ($\lambda \in [0.01, 20.0]$).
- [ ] Replicate the 2D optimization sensitivity heatmaps comparing AdamW's and Lion's optimal zones.
- [ ] Run batch-size sensitivity experiments ($N \in \{64, 256, 1024, 4096\}$) to check if the Lion-AdamW generalization gap behaves as described.

### **Week 3: Loss Geometry Flatness & Optimization Audits**
- [ ] Measure step-wise system training runtimes and check for the predicted **2–15% step-rate improvement** on GPU hardware.
- [ ] Perform a loss landscape flatness check: perturb converged model weights with random Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$.
- [ ] Calculate $L^N_{\text{train}} = \mathbb{E}_{\epsilon}[L_{\text{train}}(w + \epsilon)]$ to mathematically verify that Lion converges to a flatter, more robust minimum than AdamW.

---

## 📦 References
- **Paper**: Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V. Le. *Symbolic Discovery of Optimization Algorithms*. arXiv:2302.06675 [cs.LG], 2023.
