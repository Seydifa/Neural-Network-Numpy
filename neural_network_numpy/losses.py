import numpy as np

def log_loss(y_pred, y_true, eps=1e-8):
    """Log loss (Binary Cross-Entropy) function."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def dlog_loss(y_pred, y_true, eps=1e-8):
    """Derivative of the log loss function."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]

def mse(y_pred, y_true):
    """Mean Squared Error (MSE) loss function."""
    return np.mean(np.square(y_pred - y_true))

def dmse(y_pred, y_true):
    """Derivative of the Mean Squared Error (MSE) function."""
    return 2 * (y_pred - y_true) / y_true.size
