
# Neural Network Implementation Overview

This repo outlines the implementation details of various neural network components, including layers, activation functions, loss functions, and optimization techniques. Each section provides a brief explanation, mathematical formulations, and implementation guidance.

## Table of Contents

- [1. The Big Picture](#1-the-big-picture)
- [2. Neural Network Layers](#2-neural-network-layers)
- [3. Activation Functions](#3-activation-functions)
- [4. Neural Network Models](#4-neural-network-models)
- [5. Criterion - Loss Functions](#5-criterion---loss-functions)
- [6. Optimizers](#6-optimizers)
- [7. Regularization Techniques](#7-regularization-techniques)

## 1. The Big Picture

Neural networks function as mathematical models that map input data `x` to an output `y` using a series of nested functions representing layers:
`y = f_NN(x) = f3(f2(f1(x)))`


This approach allows the model to learn complex patterns in data, essential for tasks like spam email detection.

## 2. Neural Network Layers

### 2.1 Linear Layers

Linear layers, or fully-connected layers, connect each input neuron to every output neuron. The `Linear` class in `linear.py` involves:

- **Attributes:**
  - Learnable parameters: weights (W) and biases (b).
  - Forward propagation storage: input matrix (A), batch size (N).
  - Backward propagation storage: gradients (dLdW, dLdb).

- **Methods:**
  - `__init__(self, in_features, out_features)`
  - `forward(self, A)`
  - `backward(self, dLdZ)`

Mathematically, the linear transformation in a layer is represented as:

`Z = A * W^T + 1_N * b`



where `1_N` is a vector of ones for broadcasting the bias.

## 3. Activation Functions

Activation functions introduce non-linearity into the network, crucial for learning complex patterns. Common functions include:

- **Sigmoid**
- **ReLU**
- **Tanh**

Each function is implemented in a class with methods for both forward and backward propagation, e.g., `Sigmoid.forward(z)` and `Sigmoid.backward()`.

## 4. Neural Network Models

Models are composed of multiple layers and activation functions. Each model class, e.g., `Model`, `MLP0`, `MLP1`, `MLP4`, includes:

- **Attributes:**
  - List of layers and activation functions.

- **Methods:**
  - `forward(self, A0)`
  - `backward(self, dLdAl)`

The classes implement the forward and backward propagation for the entire network model.

## 5. Criterion - Loss Functions

Loss functions measure the discrepancy between the model output and the target values, guiding the training process. Examples include:

- **MSE (Mean Squared Error) Loss**
- **Cross-Entropy Loss**

These functions are implemented with forward and backward methods to integrate into the training loop.

## 6. Optimizers

Optimizers adjust the parameters of the network based on the gradients computed during backpropagation. A common optimizer is SGD (Stochastic Gradient Descent), implemented with options for momentum.

## 7. Regularization Techniques

Regularization methods, such as Batch Normalization, improve training stability and performance. The `BatchNorm1d` class includes:

- **Attributes:**
  - Running mean and variance.
  - Learnable scaling and shifting parameters.

- **Methods:**
  - `forward(self, Z, eval=False)`
  - `backward(self, dLdBZ)`

Regularization techniques are essential for training deep neural networks effectively.

---

For more detailed implementation and specific code examples, refer to the respective `.py` files in the project repository.

