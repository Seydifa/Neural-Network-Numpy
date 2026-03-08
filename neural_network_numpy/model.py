import numpy as np
import time
from .layer import Layer
from .activations import sigmoid, dsigmoid
from .losses import log_loss, dlog_loss
from .utils import progress_bar

class NeuralNetwork:
    def __init__(self, input_dim):
        """
        Initializes the neural network.
        
        Args:
            input_dim (int): Dimension of input features.
        """
        self.input_dim = input_dim
        self.layers = []
        self.loss_history = []

    def add_layer(self, n_neurons, activation=sigmoid, dactivation=dsigmoid):
        """Adds a layer to the network."""
        new_layer = Layer(
            n_neurons=n_neurons,
            activation=activation,
            dactivation=dactivation,
            input_dim=self.input_dim
        )
        self.layers.append(new_layer)
        self.input_dim = n_neurons

    def forward(self, x):
        """Performs a forward pass through all layers."""
        a = x
        for layer in self.layers:
            a = layer(a)
        return a

    def __call__(self, x):
        return self.forward(x)

    def backward(self, x, y, dloss_fn):
        """Performs backpropagation to compute gradients."""
        a_final = self.forward(x)
        
        # Error for the last layer
        last_layer = self.layers[-1]
        last_layer.delta = dloss_fn(a_final, y) * last_layer.dactivation(last_layer.z)
        
        # Propagate error backward
        for i in reversed(range(len(self.layers) - 1)):
            current_layer = self.layers[i]
            next_layer = self.layers[i+1]
            current_layer.delta = (next_layer.delta @ next_layer.w) * current_layer.dactivation(current_layer.z)
        
        # Compute gradients (dW, db)
        prev_a = x
        for layer in self.layers:
            layer.dw = (layer.delta.T @ prev_a)
            layer.db = layer.delta.sum(axis=0)
            prev_a = layer.a

    def get_batches(self, n_samples, batch_size, shuffle=True):
        """Generates indices for batches."""
        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)
        
        for i in range(0, n_samples, batch_size):
            yield indices[i:i + batch_size]

    def fit(self, x, y, epochs=1, lr=0.02, beta=0.9, loss_fn=log_loss, dloss_fn=dlog_loss, batch_size=None, verbose=1, shuffle=True):
        """Trains the neural network."""
        x = np.array(x)
        y = np.array(y).reshape(-1, self.layers[-1].w.shape[0])
        
        if batch_size is None or batch_size > y.shape[0]:
            batch_size = y.shape[0]
            
        n_batches = int(np.ceil(y.shape[0] / batch_size))
        
        if verbose:
            print("="*70)
            print(f"| {'Epoch':<10} | {'Loss':<20} | {'Time (s)':<20} |")
            print("="*70)
            
        start_total = time.time()
        
        for e in range(epochs):
            epoch_losses = []
            for b_idx, batch_indices in enumerate(self.get_batches(y.shape[0], batch_size, shuffle)):
                batch_x = x[batch_indices]
                batch_y = y[batch_indices]
                
                # Forward & Backward
                self.backward(batch_x, batch_y, dloss_fn)
                
                # Get current global loss for tracking (typically on full or current batch)
                # Here we use the loss on the current batch for efficiency
                pred = self.forward(batch_x)
                loss_val = loss_fn(pred, batch_y)
                epoch_losses.append(loss_val)
                
                # Update parameters
                for layer in self.layers:
                    layer.apply_rmsprop(lr, beta, layer.dw, layer.db)
                
                if not verbose:
                    progress_bar(epochs, e + 1, loss_val, b_idx + 1, n_batches)
            
            avg_loss = np.mean(epoch_losses)
            self.loss_history.append(avg_loss)
            
            if verbose:
                print(f"| {e+1:<10} | {avg_loss:<20.6f} | {time.time() - start_total:<20.6f} |")
                
        if verbose:
            print("="*70)
            print(f"{'Training Completed!':>50}")

    def summary(self):
        """Prints a summary of the network architecture."""
        print("="*70)
        print(f"| {'Multi-Layer Perceptron Summary':^66} |")
        print("="*70)
        print(f"| {'Layer':<20} | {'Neurons':<20} | {'Parameters':<20} |")
        print("-"*70)
        
        total_params = 0
        total_neurons = 0
        
        for i, layer in enumerate(self.layers):
            n_neurons = layer.w.shape[0]
            n_params = layer.w.size + layer.b.size
            print(f"| {f'Layer_{i}':<20} | {n_neurons:<20} | {n_params:<20} |")
            print("-"*70)
            total_params += n_params
            total_neurons += n_neurons
            
        print(f"| {'Total Parameters':<20} : {total_params:<43} |")
        print(f"| {'Total Neurons':<20} : {total_neurons:<43} |")
        print("="*70)
