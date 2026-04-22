# Self-Pruning Neural Network

This project implements a neural network that learns to remove unnecessary weights during training using learnable gates.

## Idea
Each weight has a gate (0–1):
- Gate ≈ 0 → weight removed
- Gate ≈ 1 → weight active

## Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|-------------|
| 0.0001 | 33.79       | 0.03        |
| 0.001  | 32.33       | 0.04        |
| 0.01   | 31.14       | 0.04        |

## Tech Stack
- Python
- PyTorch

## How to Run
pip install torch torchvision  
python main.py
