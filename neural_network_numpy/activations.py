import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    """Derivative of the sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """Rectified Linear Unit (ReLU) activation function."""
    return np.maximum(0, x)

def drelu(x):
    """Derivative of the ReLU function."""
    return np.where(x <= 0, 0, 1)

def tanh(x):
    """Hyperbolic tangent activation function."""
    return np.tanh(x)

def dtanh(x):
    """Derivative of the hyperbolic tangent function."""
    return 1 - np.tanh(x)**2

def linear(x):
    """Linear activation function."""
    return x

def dlinear(x):
    """Derivative of the linear function."""
    return np.ones_like(x)

def numerical_derivative(f, x, h=1e-9):
    """Centred Euler approximation of the derivative."""
    return (f(x + h) - f(x - h)) / (2 * h)

def softmax(x):
    """Softmax activation function."""
    exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

def dsoftmax(x):
    """
    Derivative of Softmax. 
    Note: In backprop, it's usually combined with Cross-Entropy.
    We return 1 here so that delta = (y_pred - y_true) in backward.
    """
    return 1
