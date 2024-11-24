from data.get_mnist import get_mnist
import numpy as np
import matplotlib.pyplot as plt

# Fetch the dataset
images, labels = get_mnist()

# Initialize weights and biases
w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))

# Hyperparameters
learn_rate = 0.01
epochs = 3

# Training the neural network
for epoch in range(epochs):
    nr_correct = 0
    for img, label in zip(images, labels):
        img.shape += (1,)
        label.shape += (1,)

        # Forward propagation: Input -> Hidden
        h_pre = b_i_h + w_i_h @ img
        h = 1 / (1 + np.exp(-h_pre))

        # Forward propagation: Hidden -> Output
        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        # Calculate error
        nr_correct += int(np.argmax(o) == np.argmax(label))

        # Backpropagation
        delta_o = o - label
        w_h_o -= learn_rate * delta_o @ h.T
        b_h_o -= learn_rate * delta_o

        delta_h = w_h_o.T @ delta_o * (h * (1 - h))
        w_i_h -= learn_rate * delta_h @ img.T
        b_i_h -= learn_rate * delta_h

    # Print accuracy for the epoch
    accuracy = (nr_correct / images.shape[0]) * 100
    print(f"Epoch {epoch + 1}/{epochs}, Accuracy: {accuracy:.2f}%")

# Testing the neural network
while True:
    index = int(input("Enter an index (0-59999) to test: "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap="Greys")

    img.shape += (1,)
    h_pre = b_i_h + w_i_h @ img
    h = 1 / (1 + np.exp(-h_pre))
    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"Prediction: {np.argmax(o)}")
    plt.show()
