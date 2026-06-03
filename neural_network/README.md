# Neural Network — MNIST classifier (from scratch)

This folder contains a from-scratch neural network project built with **Python + NumPy** to classify handwritten digits from the MNIST dataset.

Like the other projects in Scratchers, the goal here is learning by implementing the core ideas manually instead of relying on high-level frameworks.

## What this project covers

- Manual forward propagation
- Manual backpropagation
- Gradient descent training loop
- Multi-class classification with Softmax
- Basic evaluation on MNIST

## Architecture

The notebook implements a feed-forward neural network with this layout:

- **Input layer:** 784 neurons (28×28 image flattened)
- **Hidden layer 1:** 64 neurons, ReLU activation
- **Hidden layer 2:** 32 neurons, ReLU activation
- **Output layer:** 10 neurons (digits 0–9), Softmax activation

This keeps the model simple enough to study while still being strong enough to learn meaningful digit patterns.

## Requirements

Install the same packages used in the notebook:

- NumPy
- Pandas
- Matplotlib

Example:

```bash
pip install numpy pandas matplotlib
```

## Running it

1. Open the notebook:

```bash
jupyter notebook neural-network-from-scratch.ipynb
```

2. Run cells top-to-bottom to:
   - load and inspect data,
   - initialize parameters,
   - train with gradient descent,
   - evaluate predictions.

## Notes in the Scratchers context

This is a notebook-first learning module that fits the Scratchers theme: understanding systems deeply by building them from scratch.

If you want to extend it next, good directions are:

- modularizing notebook code into `.py` files,
- trying different layer sizes/learning rates,
- adding regularization and better evaluation tracking.
