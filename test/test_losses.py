import numpy as np
import pytest
from neural_network_numpy.losses import mse, dmse, log_loss

def test_mse():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3])
    assert mse(y_pred, y_true) == 0
    
    y_pred = np.array([2, 3, 4])
    # (1^2 + 1^2 + 1^2) / 3 = 1
    assert mse(y_pred, y_true) == 1

def test_dmse():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([2, 3, 4])
    # 2 * (y_pred - y_true) / size = 2 * [1, 1, 1] / 3 = [2/3, 2/3, 2/3]
    derivative = dmse(y_pred, y_true)
    assert np.allclose(derivative, [2/3, 2/3, 2/3])

def test_log_loss():
    y_true = np.array([1, 0])
    y_pred = np.array([0.9, 0.1])
    loss = log_loss(y_pred, y_true)
    assert loss > 0
    
    y_pred_perfect = np.array([1.0, 0.0])
    loss_perfect = log_loss(y_pred_perfect, y_true)
    assert np.isclose(loss_perfect, 0, atol=1e-7)
