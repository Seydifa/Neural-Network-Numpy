import numpy as np
from .activations import numerical_derivative

class Layer:
    def __init__(self, n_neurons, activation, dactivation=None, input_dim=2):
        """
        Initializes a layer.
        
        Args:
            n_neurons (int): Number of neurons in the layer.
            activation (callable): Activation function.
            dactivation (callable, optional): Derivative of the activation function. Defaults to numerical derivative.
            input_dim (int): Dimension of input features. Defaults to 2.
        """
        self.activation = activation
        self.dactivation = dactivation if dactivation is not None else lambda x: numerical_derivative(activation, x)
        
        # Initialize weights and biases (Xavier/He initialization style)
        self.w = (np.random.randn(n_neurons, input_dim) / np.sqrt(input_dim)).astype(np.float64)
        self.b = np.zeros(n_neurons).astype(np.float64)
        
        self.z = None # Linear part
        self.a = None # Activation part
        
        # RMSprop parameters
        self.vdw = np.zeros_like(self.w)
        self.vdb = np.zeros_like(self.b)
        
        # Gradients
        self.dw = None
        self.db = None
        self.delta = None

    def __call__(self, x):
        """Forward pass for the layer."""
        self.z = x @ self.w.T + self.b
        self.a = self.activation(self.z)
        return self.a

    def apply_rmsprop(self, lr, beta, dw, db, epsilon=1e-10):
        """Updates weights and biases using RMSprop."""
        dw = dw.reshape(self.w.shape)
        db = db.reshape(self.b.shape)
        
        self.vdw = beta * self.vdw + (1 - beta) * np.power(dw, 2)
        self.vdb = beta * self.vdb + (1 - beta) * np.power(db, 2)
        
        self.w -= (lr / np.sqrt(self.vdw + epsilon)) * dw
        self.b -= (lr / np.sqrt(self.vdb + epsilon)) * db
