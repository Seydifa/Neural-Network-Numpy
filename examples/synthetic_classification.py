import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import datasets
import os
import sys

# Add root to path to import our module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_network_numpy.model import NeuralNetwork
from neural_network_numpy.activations import sigmoid, dsigmoid
from neural_network_numpy.losses import log_loss, dlog_loss
from neural_network_numpy.utils import plot_decision_boundary

def train_and_plot(X, y, title, filename):
    y_cat = y.reshape(-1, 1)
    
    # Model architecture as defined in PDF example
    modele = NeuralNetwork(input_dim=2)
    modele.add_layer(45, activation=sigmoid, dactivation=dsigmoid)
    modele.add_layer(1, activation=sigmoid, dactivation=dsigmoid)

    print(f"Training {title}...")
    modele.fit(X, y_cat, epochs=5000, lr=0.05, loss_fn=log_loss, dloss_fn=dlog_loss, verbose=0)

    # Manual plot for saving
    min_x, min_y = X.min(axis=0) - 0.5, X.min(axis=0)[1] - 0.5
    max_x, max_y = X.max(axis=0) + 0.5, X.max(axis=0)[1] + 0.5
    
    xx, yy = np.meshgrid(np.linspace(min_x, max_x, 50), np.linspace(min_y, max_y, 50))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    predictions = modele.forward(grid_points)
    zz = predictions.reshape(xx.shape)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], marker=dict(color=y, size=10), mode='markers', name='Data'))
    fig.add_trace(go.Contour(x=np.linspace(min_x, max_x, 50), y=np.linspace(min_y, max_y, 50), z=zz, opacity=0.5, showscale=False, name='Boundary'))
    
    fig.update_layout(template='plotly_white', title=title)
    
    os.makedirs('figures', exist_ok=True)
    try:
        fig.write_image(f"figures/{filename}.png")
        print(f"Figure saved to figures/{filename}.png")
    except Exception:
        # Fallback to matplotlib
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, zz, alpha=0.5)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
        plt.title(title)
        plt.savefig(f"figures/{filename}.png")
        print(f"Figure saved to figures/{filename}.png using matplotlib")

def main():
    # 1. Moons
    X_m, y_m = datasets.make_moons(n_samples=200, noise=0.2)
    train_and_plot(X_m, y_m, "Moons Classification", "classification_moons")

    # 2. Circles
    X_c, y_c = datasets.make_circles(n_samples=200, noise=0.1)
    train_and_plot(X_c, y_c, "Circles Classification", "classification_circles")

if __name__ == "__main__":
    main()
