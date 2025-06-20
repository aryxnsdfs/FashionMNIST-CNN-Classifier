 FashionMNIST-CNN-Classifier

A Convolutional Neural Network (CNN) model built using TensorFlow and Keras to classify images from the Fashion MNIST dataset. This project demonstrates the full pipeline of data preprocessing, model creation, training, evaluation, and performance visualization.

---

## 📌 Project Overview

The Fashion MNIST dataset consists of 28x28 grayscale images of 10 clothing categories. This project uses a deep CNN model to classify these images with high accuracy.

---

## 🧠 Model Architecture

* **Conv2D (32 filters, 3x3, ReLU)**
* **MaxPooling2D (2x2)**
* **Conv2D (64 filters, 3x3, ReLU)**
* **MaxPooling2D (2x2)**
* **Conv2D (64 filters, 3x3, ReLU)**
* **Flatten**
* **Dense (64 units, ReLU)**
* **Dense (10 units, Softmax)**

---

## 📊 Results

The model is trained for 10 epochs and achieves high accuracy on both training and test datasets. Accuracy and loss curves are plotted to show the model's learning progress.

---

## 🔧 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/FashionMNIST-CNN-Classifier.git
   cd FashionMNIST-CNN-Classifier
   ```

2. Install dependencies:

   ```bash
   pip install tensorflow matplotlib
   ```

---

## 🚀 Run the Code

```bash
python fashion_mnist_cnn.py
```

---

## 📈 Visualizations

The script displays plots of:

* Training vs Validation Accuracy
* Training vs Validation Loss

These help to evaluate how well the model generalizes.

---

## 📁 Dataset

Fashion MNIST is automatically downloaded via TensorFlow's datasets API:

```python
fashion_mnist = tf.keras.datasets.fashion_mnist
```

---

## 📜 License

This project is licensed under the MIT License.

---

