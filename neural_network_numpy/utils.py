import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def progress_bar(total, progress, loss, batch, nb_batch):
    """Displays a progress bar for training."""
    bar_length = 50
    status = ""
    progress_ratio = float(progress) / float(total)
    
    if progress_ratio >= 1.0:
        progress_ratio = 1.0
        status = "\r\n"
        
    block = int(round(bar_length * progress_ratio))
    text = "\r[{}] {:.1f}% {} Loss = {:<10.6f} Epochs = {}/{} Batch = {}/{} ".format(
        "=" * block + "-" * (bar_length - block),
        round(progress_ratio * 100, 0),
        status,
        loss,
        progress,
        total,
        batch,
        nb_batch
    )
    sys.stdout.write(text)
    sys.stdout.flush()

def plot_decision_boundary(model, X, y, height=500, width=500, return_fig=False):
    """Plots the decision boundary for 2D classification problems."""
    min_x, min_y = X.min(axis=0) - 0.5, X.min(axis=0)[1] - 0.5
    max_x, max_y = X.max(axis=0) + 0.5, X.max(axis=0)[1] + 0.5
    
    xx, yy = np.meshgrid(np.linspace(min_x, max_x, 50), np.linspace(min_y, max_y, 50))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    predictions = model.forward(grid_points)
    zz = predictions.reshape(xx.shape)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], marker=dict(color=y, size=10), mode='markers'))
    fig.add_trace(go.Contour(x=np.linspace(min_x, max_x, 50), y=np.linspace(min_y, max_y, 50), z=zz, opacity=0.5, showscale=False))
    
    fig.update_layout(height=height, width=width, template='plotly_white')
    
    if return_fig:
        return fig
    fig.show()

def to_categorical(y, num_classes=None):
    """One-hot encodes labels."""
    y = np.array(y, dtype='int')
    if not num_classes:
        num_classes = np.max(y) + 1
    res = np.zeros((y.size, num_classes))
    res[np.arange(y.size), y] = 1
    return res
