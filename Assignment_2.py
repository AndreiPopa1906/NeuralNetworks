import numpy as np
from torchvision.datasets import MNIST
import time

def download_mnist(is_train: bool):
    dataset = MNIST(root='./data',
                    transform=lambda x: np.array(x).flatten() / 255.0,
                    download=True,
                    train=is_train)
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)

    return mnist_data, mnist_labels


def one_hot_encode(labels: list, num_classes: int = 10):
    one_hot_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        one_hot_labels[i, label] = 1
    return one_hot_labels


def softmax(x: np.array):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_true: np.array, y_pred: np.array):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))  # for numeric stability (avoid log(0))


class Perceptron():
    def __init__(self, input_size: int, num_classes: int):
        self.weights = np.random.rand(input_size, num_classes) * 0.01
        self.bias = np.random.rand(num_classes) * 0.01

    def forward(self, x: np.array):
        z = np.dot(x, self.weights) + self.bias
        return softmax(z)

    def train(self, x: np.array, y: np.array, learning_rate: float):
        y_pred = self.forward(x)
        gradient = y_pred - y

        # Update weights and bias
        self.weights -= learning_rate * np.dot(x.T, gradient) / x.shape[0]
        self.bias -= learning_rate * np.mean(gradient, axis=0)

        return cross_entropy_loss(y, y_pred)

    def accuracy(self, x: np.array, y: np.array):
        y_pred = self.forward(x)
        predictions = np.argmax(y_pred, axis=1)
        targets = np.argmax(y, axis=1)
        return np.mean(predictions == targets)


if __name__ == "__main__":
    # Load dataset
    mnist_data, mnist_labels = download_mnist(is_train=True)
    test_data, test_labels = download_mnist(is_train=False)

    # Prepare data
    data = np.array(mnist_data)
    labels = one_hot_encode(mnist_labels)
    test_data = np.array(test_data)
    test_labels = one_hot_encode(test_labels)

    # Model parameters
    input_size = data.shape[1]
    num_classes = 10
    learning_rate = 0.01
    epochs = 300
    batch_size = 100
    decay_coefficient = 0.95

    # Initialize model
    model = Perceptron(input_size, num_classes)

    # Training loop
    start_time = time.time()
    for epoch in range(epochs):
        # Shuffle data
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        
        epoch_loss = 0
        for i in range(0, len(data), batch_size):
            x_batch = data[i:i + batch_size]
            y_batch = labels[i:i + batch_size]

            # Train model on batch
            batch_loss = model.train(x_batch, y_batch, learning_rate)
            epoch_loss += batch_loss

        # Evaluate accuracy every 10 epochs
        if (epoch + 1) % 10 == 0:
            train_accuracy = model.accuracy(data, labels)
            test_accuracy = model.accuracy(test_data, test_labels)
            print(f"Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # diminishing the learning rate after some epochs for greater precision
        learning_rate *= decay_coefficient

        # if epoch >= 70:
        #     learning_rate = 0.005
        #
        # if epoch >= 110:
        #     learning_rate = 0.001

    # Final evaluation
    final_test_accuracy = model.accuracy(test_data, test_labels)
    print(f"Final Test Accuracy: {final_test_accuracy:.4f}")
    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.2f} seconds")
