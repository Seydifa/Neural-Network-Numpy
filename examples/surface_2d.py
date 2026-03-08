import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# Add root to path to import our module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_network_numpy.model import NeuralNetwork
from neural_network_numpy.activations import relu, drelu, linear, dlinear
from neural_network_numpy.losses import mse, dmse

def f(x, y):
    return x * np.cos(y)

def main():
    n = 25
    xmin, xmax, ymin, ymax = -4.0, 4.0, -4.0, 4.0
    VX = np.linspace(xmin, xmax, n)
    VY = np.linspace(ymin, ymax, n)
    X, Y = np.meshgrid(VX, VY)
    Z = f(X, Y)
    
    x_train = np.c_[X.ravel(), Y.ravel()]
    y_train = Z.reshape(-1, 1)

    # Model architecture as defined in PDF example
    modele = NeuralNetwork(input_dim=2)
    modele.add_layer(50, activation=relu, dactivation=drelu)
    modele.add_layer(50, activation=relu, dactivation=drelu)
    modele.add_layer(1, activation=linear, dactivation=dlinear)

    print("Training 2D Surface Approximation...")
    modele.fit(x_train, y_train, lr=0.01, epochs=5000, loss_fn=mse, dloss_fn=dmse, verbose=0, batch_size=100)

    Z_pred = modele.forward(x_train).reshape(Z.shape)

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "surface"}, {"type": "surface"}]],
                        subplot_titles=("Ground Truth Surface", "Model Estimated Surface"))
    
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, showscale=False), 1, 1)
    fig.add_trace(go.Surface(x=X, y=Y, z=Z_pred, showscale=False), 1, 2)
    
    fig.update_layout(template='plotly_white', title="2D Surface Approximation")
    
    os.makedirs('figures', exist_ok=True)
    try:
        fig.write_image("figures/surface_2d.png")
        print("Figure saved to figures/surface_2d.png")
    except Exception as e:
        print(f"Could not save image: {e}. Plotly 3D requires 'kaleido'.")
        # For 3D, matplotlib is also an option but complicated for surfaces
        import matplotlib.pyplot as plt
        from matplotlib import cm
        fig = plt.figure(figsize=(12, 5))
        
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm)
        ax1.set_title("Ground Truth")
        
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.plot_surface(X, Y, Z_pred, cmap=cm.coolwarm)
        ax2.set_title("Model Prediction")
        
        plt.savefig("figures/surface_2d.png")
        print("Figure saved to figures/surface_2d.png using matplotlib")

if __name__ == "__main__":
    main()
