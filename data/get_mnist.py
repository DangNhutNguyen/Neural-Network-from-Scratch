import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder

def get_mnist():
    # Fetch MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data / 255.0, mnist.target.astype(int)

    # One-hot encode labels
    encoder = OneHotEncoder(sparse_output=False)
    y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

    return X.astype(np.float32), y_one_hot.astype(np.float32)
