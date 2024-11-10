import os
import numpy as np
from torchvision.datasets import MNIST
import time

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]


def download_mnist(is_train: bool):
    if not os.path.exists('./data'):
        dataset = MNIST(root='./data',
                        transform=lambda x: np.array(x).flatten() / 255.0,
                        download=True,
                        train=is_train)
    else:
        dataset = MNIST(root='./data',
                        transform=lambda x: np.array(x).flatten() / 255.0,
                        download=False,
                        train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)

    return np.array(mnist_data), np.array(mnist_labels)


class Layer():
    def __init__(self, input_size, output_size, dropout_rate=0.0):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))
        self.bias = np.zeros((1, output_size))
        self.learning_rate = 0.5
        self.dropout_rate = dropout_rate
        self.dropout_mask = None

    def forward(self, x: np.array, use_dropout=True):
        self.input = x
        self.z = np.dot(x, self.weights) + self.bias
        self.output = sigmoid(self.z)

        if use_dropout and self.dropout_rate > 0.0:
            self.dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, size=self.output.shape) / (
                        1 - self.dropout_rate)
            self.output *= self.dropout_mask

        return self.output

    def backward(self, error: np.array):
        if self.dropout_rate > 0.0 and self.dropout_mask is not None:
            error *= self.dropout_mask
        delta = error * sigmoid_derivative(self.output)
        dW = np.dot(self.input.T, delta) / self.input.shape[0]
        dB = np.sum(delta, axis=0, keepdims=True) / self.input.shape[0]
        self.weights -= self.learning_rate * dW
        self.bias -= self.learning_rate * dB
        return np.dot(delta, self.weights.T)

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate


class NeuralNetwork():
    def __init__(self):
        self.layers = []

    def add_layer(self, input_size, output_size, dropout_rate=0.0):
        layer = Layer(input_size, output_size, dropout_rate)
        self.layers.append(layer)

    def forward(self, x: np.array, use_dropout=True):
        for layer in self.layers:
            x = layer.forward(x, use_dropout=use_dropout)
        return x

    def backward(self, error: np.array):
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def fit(self, X, y, X_val, y_val, epochs=10, batch_size=64, learning_rate=0.99, patience=5, decay_factor=0.5):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            mean_loss = 0
            for i in range(0, X.shape[0], batch_size):
                x_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                y_pred = self.forward(x_batch, use_dropout=True)
                error = y_pred - y_batch
                self.backward(error)
                loss = cross_entropy_loss(y_batch, y_pred)
                mean_loss += loss
            mean_loss /= (X.shape[0] / batch_size)
            train_accuracy = self.evaluate(X, y)
            val_accuracy = self.evaluate(X_val, y_val)
            val_loss = cross_entropy_loss(y_val, self.forward(X_val, use_dropout=False))
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {mean_loss:.4f}, Training Accuracy: {train_accuracy * 100:.2f}%, Validation Accuracy: {val_accuracy * 100:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    learning_rate *= decay_factor
                    for layer in self.layers:
                        layer.set_learning_rate(learning_rate)
                    print(f"Learning rate reduced to {learning_rate:.6f} due to plateau in validation loss.")
                    patience_counter = 0

    def evaluate(self, X, y):
        y_pred = self.forward(X, use_dropout=False)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y, axis=1)
        accuracy = np.sum(y_pred_labels == y_true_labels) / y.shape[0]
        return accuracy


def main():
    start_time = time.time()
    X_train, y_train = download_mnist(is_train=True)
    X_test, y_test = download_mnist(is_train=False)
    y_train_one_hot = np.eye(10)[y_train]
    y_test_one_hot = np.eye(10)[y_test]
    split_idx = int(0.85 * X_train.shape[0])
    X_train, X_val = X_train[:split_idx], X_train[split_idx:]
    y_train, y_val = y_train_one_hot[:split_idx], y_train_one_hot[split_idx:]
    nn = NeuralNetwork()
    nn.add_layer(input_size=784, output_size=100, dropout_rate=0.1)
    nn.add_layer(input_size=100, output_size=10)
    nn.fit(X_train, y_train, X_val, y_val, epochs=100, batch_size=64, learning_rate=0.05, patience=10, decay_factor=0.8)
    test_accuracy = nn.evaluate(X_test, y_test_one_hot)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
