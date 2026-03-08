import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objects as go

# Add path to import local module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_network_numpy import NeuralNetwork
from neural_network_numpy.activations import sigmoid, relu, softmax
from neural_network_numpy.losses import log_loss, categorical_cross_entropy, dcategorical_cross_entropy

def main():
    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)

    # --- 1. XOR Logic Gate ---
    print("\n--- 1. XOR Logic Gate ---")
    X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
    Y_xor = np.array([[0], [1], [1], [0]])

    modele_xor = NeuralNetwork(input_dim=2)
    modele_xor.add_layer(4, activation=sigmoid)
    modele_xor.add_layer(1, activation=sigmoid)

    print("Training XOR...")
    modele_xor.fit(X_xor, Y_xor, epochs=5000, lr=0.5, verbose=0, shuffle=False)

    print("XOR Predictions:")
    preds = modele_xor.forward(X_xor)
    for i in range(len(X_xor)):
        print(f"Input: {X_xor[i]} -> Target: {Y_xor[i][0]} -> Predicted: {preds[i][0]:.4f}")

    # Generate XOR Decision Boundary Figure
    h = .02
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = modele_xor.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X_xor[:, 0], X_xor[:, 1], c=Y_xor.ravel(), s=100, edgecolors='k', cmap=plt.cm.Spectral)
    plt.title("XOR Decision Boundary")
    plt.savefig("figures/xor_boundary.png")
    print("Saved: figures/xor_boundary.png")

    # --- 2. MNIST Digits Classification ---
    print("\n--- 2. MNIST Digits Classification ---")
    print("Loading MNIST (subset of 10000)...")
    X_mnist, y_mnist = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    X_mnist = X_mnist[:10000] / 255.0  # Subset for speed
    y_mnist = y_mnist[:10000].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size=0.2, random_state=42)

    encoder = OneHotEncoder(sparse_output=False)
    y_train_cat = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_cat = encoder.transform(y_test.reshape(-1, 1))

    # Our Custom Neural Network
    nn = NeuralNetwork(input_dim=784)
    nn.add_layer(128, activation=relu)
    nn.add_layer(10, activation=softmax)

    print("Training Custom NumPy Model...")
    nn.fit(X_train, y_train_cat, epochs=30, lr=0.01, batch_size=128, 
           loss_fn=categorical_cross_entropy, dloss_fn=dcategorical_cross_entropy, verbose=1)

    y_pred_probs = nn.forward(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    custom_accuracy = np.mean(y_pred == y_test)
    print(f"Custom Model Accuracy: {custom_accuracy*100:.2f}%")

    # --- 3. Scikit-Learn Comparison ---
    print("\nTraining Scikit-Learn MLPClassifier for Comparison...")
    mlp = MLPClassifier(hidden_layer_sizes=(128,), activation='relu', solver='sgd', 
                        learning_rate_init=0.01, max_iter=30, batch_size=128)
    mlp.fit(X_train, y_train)

    sklearn_accuracy = mlp.score(X_test, y_test)
    print(f"Sklearn Model Accuracy: {sklearn_accuracy*100:.2f}%")

    # Save Comparison Figure
    plt.figure(figsize=(10, 6))
    plt.plot(nn.loss_history, label=f'Custom NumPy (Acc: {custom_accuracy:.2%})', linewidth=2)
    plt.plot(mlp.loss_curve_, label=f'Scikit-Learn (Acc: {sklearn_accuracy:.2%})', linestyle='--')
    plt.title('MNIST Training Loss: Custom vs Scikit-Learn')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("figures/mnist_benchmark.png")
    print("Saved: figures/mnist_benchmark.png")

if __name__ == "__main__":
    main()
