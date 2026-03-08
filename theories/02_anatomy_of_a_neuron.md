# 02 - Anatomy of a Artificial Neuron

Before building a complex network, we must understand its most basic unit: the **Artificial Neuron** (inspired by the biological neurons in our brains).

## The Biological Analogy

Think of a neuron as a tiny factory worker:
1. **Inputs (Dendrites)**: They receive signals from other workers.
2. **Weights (Importance)**: Some signals are more important than others. The worker multiplies each signal by a "weight" to reflect its significance.
3. **Bias (Threshold)**: The worker has a personal threshold. Even if signals are coming in, they won't pass them on unless the total "energy" exceeds a certain level.
4. **Activation Function (The Decision)**: If the total energy is high enough, the worker sends a signal down the line.

## The Mathematical Formulation

Mathematically, a single neuron performs a two-step process:

### 1. The Linear Aggregation (Z)

We take all inputs ($x_1, x_2, \dots, x_n$), multiply them by their respective weights ($w_1, w_2, \dots, w_n$), and add a bias ($b$).

$$z = (w_1 \cdot x_1 + w_2 \cdot x_2 + \dots + w_n \cdot x_n) + b$$

In matrix notation, if $X$ is our input vector and $W$ is our weight vector:
$$z = X \cdot W^T + b$$

> [!TIP]
> **The Bias ($b$):** Imagine you are trying to decide if you should go outside. The bias is like your "initial mood". If you are naturally lazy ($b$ is very negative), you need a lot of positive signals (sunshine, friends calling) to actually move.

### 2. The Non-Linear Activation (A)

If we only used linear math, our network would just be one giant linear function—incapable of solving complex, curvy problems. We need to introduce "texture" or "non-linearity".

We pass $z$ through an **Activation Function** $\sigma$:

$$a = \sigma(z)$$

The result $a$ (Activation) is what this neuron sends to the next layer.

---

In the next section, we will explore the different types of **[Activation Functions](03_activation_functions.md)** and why choosing the right one is critical.
