# Neural Network from Scratch

This project demonstrates the implementation of a simple feedforward neural network from scratch using only NumPy. The neural network is trained on the MNIST dataset, which consists of handwritten digits. The network is trained using gradient descent and backpropagation.

## Mathematics and Algorithms

### 1. **Feedforward Propagation**

In the **feedforward step**, we calculate the output of the hidden and output layers.

#### **Hidden Layer Calculation:**

The pre-activation value for the hidden layer is computed as:

$$
\mathbf{h_{pre}} = W_{i \to h} \cdot \text{img} + b_{i \to h}
$$

Where:
- $$\( W_{i \to h} \)$$ is the weight matrix from the input layer to the hidden layer.
- $$\( \text{img} \)$$ is the input image vector.
- $$\( b_{i \to h} \)$$ is the bias term for the hidden layer.

Then, we apply the **sigmoid activation function** to the result:

$$
\mathbf{h} = \sigma(\mathbf{h_{pre}}) = \frac{1}{1 + e^{-\mathbf{h_{pre}}}}
$$

Where:
- $$\( \sigma(x) = \frac{1}{1 + e^{-x}} \)$$ is the sigmoid function.

#### **Output Layer Calculation:**

The pre-activation value for the output layer is computed as:

$$
\mathbf{o_{pre}} = W_{h \to o} \cdot \mathbf{h} + b_{h \to o}
$$

Where:
- $$\( W_{h \to o} \)$$ is the weight matrix from the hidden layer to the output layer.
- $$\( \mathbf{h} \)$$ is the hidden layer output.
- $$\( b_{h \to o} \)$$ is the bias term for the output layer.

Again, we apply the **sigmoid activation function** to get the output:

$$
\mathbf{o} = \sigma(\mathbf{o_{pre}}) = \frac{1}{1 + e^{-\mathbf{o_{pre}}}}
$$

Where:
- $$\( \mathbf{o} \)$$ is the network's final output representing the probability distribution over the possible digit classes.

### 2. **Backpropagation**

#### **Output Layer Error:**

The error for the output layer is computed as the difference between the predicted output \( \mathbf{o} \) and the actual label:

$$
\delta_o = \mathbf{o} - \text{label}
$$

#### **Update Weights and Biases for Output Layer:**

The weights and biases of the output layer are updated using the error gradient concerning those parameters. The weight update rule is:

$$
W_{h \to o} \leftarrow W_{h \to o} - \eta \cdot \delta_o \cdot \mathbf{h}^T
$$

Where $$\( \eta \)$$ is the learning rate.

The bias update rule is:

$$
b_{h \to o} \leftarrow b_{h \to o} - \eta \cdot \delta_o
$$

#### **Hidden Layer Error:**

The error for the hidden layer is propagated backward using the weights and the error term from the output layer:

$$
\delta_h = \left( W_{h \to o}^T \cdot \delta_o \right) \cdot \mathbf{h} \cdot (1 - \mathbf{h})
$$

#### **Update Weights and Biases for Hidden Layer:**

The weights and biases of the hidden layer are updated using the gradient of the error with respect to those parameters. The weight update rule is:

$$
W_{i \to h} \leftarrow W_{i \to h} - \eta \cdot \delta_h \cdot \text{img}^T
$$

And the bias update rule is:

$$
b_{i \to h} \leftarrow b_{i \to h} - \eta \cdot \delta_h
$$

### 3. **Accuracy Calculation**

Once the neural network has made predictions, the accuracy is calculated as:

$$
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Samples}} \times 100
$$

Where:
- **Number of Correct Predictions** is the count of predictions where the predicted label matches the true label.
- **Total Samples** is the total number of images in the dataset.

---

## How to Run the Code

### Prerequisites

Ensure that you have Python 3.x installed, along with the required libraries. You can install them using `pip`:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Code

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dangnhutnguyen/NeuralNetworkFromScratch.git
   cd NeuralNetworkFromScratch
   ```

2. **Run the Code**:
   To train the neural network and test it, run the following command:

   ```bash
   python main.py
   ```

3. **Training**:
   The code will train the neural network on the MNIST dataset for 3 epochs, and it will output the accuracy after each epoch.

4. **Testing**:
   After training, the program will ask you to input an index (between 0 and 59999) of an image in the MNIST dataset. It will then display the image along with the predicted digit.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
