import numpy as np
import pytest
from neural_network_numpy.activations import sigmoid, relu, tanh, numerical_derivative

def test_sigmoid():
    assert sigmoid(0) == 0.5
    assert sigmoid(10) > 0.99
    assert sigmoid(-10) < 0.01

def test_relu():
    assert relu(5) == 5
    assert relu(-5) == 0
    assert relu(0) == 0

def test_tanh():
    assert np.isclose(tanh(0), 0)
    assert tanh(10) > 0.99
    assert tanh(-10) < -0.99

def test_numerical_derivative():
    f = lambda x: x**2
    # Derivative of x^2 is 2x. At x=2, it's 4.
    df = numerical_derivative(f, 2)
    assert np.isclose(df, 4, atol=1e-5)
