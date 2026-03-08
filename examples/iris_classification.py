import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import sys

# Add root to path to import our module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_network_numpy.model import NeuralNetwork
from neural_network_numpy.activations import sigmoid, dsigmoid
from neural_network_numpy.losses import log_loss, dlog_loss
from neural_network_numpy.utils import to_categorical

def main():
    # Load data
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Preprocessing as defined in PDF
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)

    # Model architecture
    modele = NeuralNetwork(input_dim=4)
    modele.add_layer(64, activation=sigmoid, dactivation=dsigmoid)
    modele.add_layer(64, activation=sigmoid, dactivation=dsigmoid)
    modele.add_layer(3, activation=sigmoid, dactivation=dsigmoid)

    print("Training Iris Classification...")
    modele.fit(X_train, y_train_cat, epochs=2000, lr=0.05, batch_size=64, verbose=0, shuffle=True)

    # Evaluation
    y_pred_test = np.argmax(modele.forward(X_test), axis=1)
    accuracy = np.mean(y_pred_test == y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")

    # Plot loss
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(modele.loss_history)
    plt.title("Iris Classification - Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Log Loss")
    plt.grid(True)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig("figures/classification_iris_loss.png")
    print("Loss plot saved to figures/classification_iris_loss.png")

if __name__ == "__main__":
    main()
