import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = load_breast_cancer()
x = data.data
y = data.target.reshape(-1, 1)  # shape: (n_samples, 1)

# Normalize features (important for neural nets!)
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split into train and test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print(f"Training samples: {x_train.shape[0]}, Features: {x_train.shape[1]}")
print(f"Test samples: {x_test.shape[0]}") 

# Activation Function
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    return a * (1 - a)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def initialize_network(layer_sizes):
    """
    layer_sizes: list like [n_features, hidden1, hidden2, ..., n_output]
    Returns a list of (weights, biases) tuples for each layer.
    """
    np.random.seed(42)
    params = []
    for i in range(len(layer_sizes) - 1):
        w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
        b = np.zeros((1, layer_sizes[i+1]))
        params.append((w, b))
    return params

def forward(X, params):
    """
    Run input X through all layers.
    Returns list of activations (needed for backprop).
    """
    activations = [X]
    for i, (w, b) in enumerate(params):
        z = activations[-1] @ w + b
        if i == len(params) - 1:
            a = sigmoid(z)       # output layer
        else:
            a = relu(z)          # hidden layer(s)
        activations.append(a)
    return activations


def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-8  # avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def backward(y, activations, params):
    """
    Compute gradients via backpropagation.
    Returns list of (dw, db) for each layer.
    """
    m = y.shape[0]
    grads = []

    # Output layer error (derivative of BCE + sigmoid combined)
    delta = activations[-1] - y

    for i in reversed(range(len(params))):
        a_prev = activations[i]
        dw = (a_prev.T @ delta) / m
        db = np.sum(delta, axis=0, keepdims=True) / m
        grads.insert(0, (dw, db))

        if i > 0:  # propagate error to previous layer
            delta = (delta @ params[i][0].T) * relu_derivative(activations[i])

    return grads

def update_params(params, grads, learning_rate):
  new_params = []
  for (w, b), (dw, db) in zip(params, grads):
    w = w - learning_rate * dw
    b = b - learning_rate * db
    new_params.append((w, b))
  return new_params

def train(x_train, y_train, layer_sizes, epochs=500, learning_rate=0.1):
    params = initialize_network(layer_sizes)
    losses = []

    for epoch in range(epochs):
      activations = forward(x_train, params)
      loss = binary_cross_entropy(y_train, activations[-1])
      losses.append(loss)
      grads = backward(y_train, activations, params) # Added activations and params to backward call
      params = update_params(params, grads, learning_rate) # Fixed typo: paramds -> params

      if epoch % 50 == 0:
        print(f"Epoch {epoch}: loss = {loss:.4f}") # Fixed typo: printf -> print and formatting

    return params, losses

n_features = x_train.shape[1]
layer_sizes = [n_features, 16, 1]

params, losses = train(x_train, y_train, layer_sizes, epochs=500, learning_rate=0.1)

def predict(X, params):
    activations = forward(X, params)
    predictions = (activations[-1] >= 0.5).astype(int)
    return predictions

y_pred = predict(x_test, params)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy * 100:.1f}%")

import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.show()

