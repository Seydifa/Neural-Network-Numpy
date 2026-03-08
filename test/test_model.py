import numpy as np
from neural_network_numpy.model import NeuralNetwork
from neural_network_numpy.activations import sigmoid, dsigmoid
from neural_network_numpy.losses import mse, dmse

def test_model_initialization():
    nn = NeuralNetwork(input_dim=2)
    nn.add_layer(4, activation=sigmoid, dactivation=dsigmoid)
    nn.add_layer(1, activation=sigmoid, dactivation=dsigmoid)
    
    assert len(nn.layers) == 2
    assert nn.layers[0].w.shape == (4, 2)
    assert nn.layers[1].w.shape == (1, 4)

def test_forward_pass():
    nn = NeuralNetwork(input_dim=2)
    nn.add_layer(2, activation=sigmoid, dactivation=dsigmoid)
    x = np.array([[1, 2], [3, 4]])
    output = nn.forward(x)
    assert output.shape == (2, 2)

def test_backward_pass():
    nn = NeuralNetwork(input_dim=2)
    nn.add_layer(2, activation=sigmoid, dactivation=dsigmoid)
    nn.add_layer(1, activation=sigmoid, dactivation=dsigmoid)
    
    x = np.array([[1, 2]])
    y = np.array([[1]])
    
    # Run backward to compute gradients
    nn.backward(x, y, dmse)
    
    assert nn.layers[0].dw is not None
    assert nn.layers[1].dw is not None
    assert nn.layers[0].dw.shape == nn.layers[0].w.shape
    assert nn.layers[1].dw.shape == nn.layers[1].w.shape
