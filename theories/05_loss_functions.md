# 05 - Loss Functions: Measuring the "Oops"

A neural network starts with random weights. Its first predictions will be terrible. To improve, it needs a way to quantify **exactly how wrong** it is. This metric is the **Loss Function**.

## Loss vs. Cost
- **Loss**: The error for a **single** data point.
- **Cost**: The **average** loss over all data points in your training set.

## Common Loss Functions

### 1. Mean Squared Error (MSE)
Used for **Regression** (predicting a continuous number, like house prices).
It takes the difference between prediction and reality, squares it (to make it positive), and averages it.

$$ \mathcal{L}(\hat{y}, y) = (\hat{y} - y)^2 $$

- **Intuition:** Large errors are punished much more severely than small errors because of the squaring.

### 2. Log Loss / Binary Cross-Entropy (BCE)
Used for **Classification** (predicting a category, like Cat vs. Dog).
It compares the predicted probability to the actual label (0 or 1).

$$ \mathcal{L}(\hat{y}, y) = -[y \ln(\hat{y}) + (1 - y) \ln(1 - \hat{y})] $$

- **Intuition:** If the real label is $1$ and you predict a low probability (e.g., $0.1$), the logarithm sends the loss through the roof ($\infty$). If you're correct, the loss is close to $0$.

---

## Why does this matter?

The Loss Function is like the "Teacher". It doesn't tell the network *how* to fix the mistake; it just points at the mistake and says "this is how bad you did". 

To fix the weights, we need **[Backpropagation](06_backpropagation.md)**.
