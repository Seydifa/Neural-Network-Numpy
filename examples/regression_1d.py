import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add root to path to import our module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_network_numpy.model import NeuralNetwork
from neural_network_numpy.activations import tanh, dtanh, linear, dlinear
from neural_network_numpy.losses import mse, dmse

def f(x):
    return np.cos(2*x) + x*np.sin(3*x) + x**0.5 - 2

def main():
    a, b = 0, 5 
    N = 100 
    X = np.linspace(a, b, N)
    Y = f(X)
    x_train = X.reshape(-1, 1)
    y_train = Y.reshape(-1, 1)

    # Model architecture as defined in PDF example
    p = 10
    modele = NeuralNetwork(input_dim=1)
    modele.add_layer(p, activation=tanh, dactivation=dtanh)
    modele.add_layer(p, activation=tanh, dactivation=dtanh)
    modele.add_layer(p, activation=tanh, dactivation=dtanh)
    modele.add_layer(p, activation=tanh, dactivation=dtanh)
    modele.add_layer(p, activation=tanh, dactivation=dtanh)
    modele.add_layer(1, activation=linear, dactivation=dlinear)

    print("Training 1D Regression...")
    modele.fit(x_train, y_train, lr=1e-3, epochs=4000, loss_fn=mse, dloss_fn=dmse, verbose=0, batch_size=100)

    Y_pred = modele.forward(x_train)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Ground Truth vs Prediction", "Loss Evolution"))
    
    fig.add_trace(go.Scatter(x=X, y=Y, name='Ground Truth'), 1, 1)
    fig.add_trace(go.Scatter(x=X, y=Y_pred.ravel(), name='Model Prediction', marker=dict(color='red'), mode='lines'), 1, 1)
    fig.add_trace(go.Scatter(y=modele.loss_history, name='Loss'), 1, 2)
    
    fig.update_layout(template='plotly_white', title="1D Function Approximation")
    
    # Save the figure
    os.makedirs('figures', exist_ok=True)
    try:
        fig.write_image("figures/regression_1d.png")
        print("Figure saved to figures/regression_1d.png")
    except Exception as e:
        print(f"Could not save image: {e}. Plotly requires 'kaleido' to save images.")
        # Fallback: using matplotlib for saving if plotly fails
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(X, Y, label='Ground Truth')
        plt.plot(X, Y_pred.ravel(), 'r--', label='Model Prediction')
        plt.legend()
        plt.title("Ground Truth vs Prediction")
        
        plt.subplot(1, 2, 2)
        plt.plot(modele.loss_history)
        plt.title("Loss Evolution")
        plt.savefig("figures/regression_1d.png")
        print("Figure saved to figures/regression_1d.png using matplotlib")

if __name__ == "__main__":
    main()
