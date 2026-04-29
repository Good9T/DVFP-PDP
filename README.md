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
### 4. Cross-Scale Generalization Experiments
Additional generalization experiments can be conducted by modifying key hyperparameters in `env_params`.
The model pre-trained on medium-scale instances (e.g., 50-node) can be directly evaluated on larger unseen problem scales to verify cross-scale transfer ability.

### 5. Configuration for 150-Node Unseen Instances
In our experiments, different problem scales adopt distinct capacity values:
- PDTSP20: capacity = 20
- PDTSP50: capacity = 30
- PDTSP100: capacity = 40

For the unseen 150-node test set (without independent training), 
we further increase the capacity to **50** for a more challenging and comprehensive generalization evaluation in our paper.
