# DVFP-PDP
Official Implementation of Dual-View Flip-Pair (DVFP)
Symmetry-Aware Neural Reinforcement Learning for Pickup-and-Delivery Routing Problems

## Introduction
This repository provides the complete code of **DVFP (Dual-View Flip-Pair)**, a novel neural combinatorial optimization framework for PDTSP and m-PDTSP.
Representative baseline methods are integrated for comprehensive comparison:
- POMO
- PDSNO
- MatrixNet

## Project Description
DVFP explicitly models the inherent symmetry and flip-pair equivalence of pickup-and-delivery tasks.
Different from conventional single-view solving paradigms, the dual-view design stabilizes training, mitigates overfitting, and improves cross-scale generalization.

## Key Notes
1. The DVFP model only supports **distance matrix input**.
2. Euclidean correlation directories are reserved exclusively for baseline methods.


## Quick Start
### 1. Hyperparameter Switching
All scale-related configurations are concentrated in `env_params` of `train.py`, with detailed annotations:
- PDTSP20：customer_size=10, node_size=21, pomo_size=10
- PDTSP50：customer_size=25, node_size=51, pomo_size=25
- PDTSP100：customer_size=50, node_size=101, pomo_size=50
- m-PDTSP20：customer_size=10, node_size=21, pomo_size=10, capacity=20
- m-PDTSP50：customer_size=25, node_size=51, pomo_size=25, capacity=30
- m-PDTSP100：customer_size=50, node_size=101, pomo_size=50, capacity=40

### 2. Run Training
```bash
python train.py
```
### 3. Run Inference
```bash
python test.py
```
