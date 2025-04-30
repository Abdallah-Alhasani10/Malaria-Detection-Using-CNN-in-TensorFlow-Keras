# 🧬 Malaria Detection Using Convolutional Neural Networks (CNN)

This project demonstrates how to detect malaria-infected cells using deep learning techniques. It trains CNN models for binary classification: **Parasitized** vs **Uninfected**.

## 🧠 Models Implemented

Three different modeling approaches in TensorFlow/Keras are demonstrated to show flexibility in building neural networks:

### 1. Sequential API
A straightforward stack of layers, ideal for quick experimentation.

```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(...),
    ...
])
```

### 2. Functional API
A more flexible way to build non-linear and multi-input/output models.

```python
inputs = tf.keras.Input(shape=(...))
x = tf.keras.layers.Conv2D(...)(inputs)
...
model = tf.keras.Model(inputs, outputs)
```

### 3. Subclassing API
Defines a custom model class by subclassing `tf.keras.Model`, giving full control over the forward pass.

```python
class MalariaCNN(tf.keras.Model):
    def __init__(self):
        super(MalariaCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(...)
        ...
    
    def call(self, inputs):
        x = self.conv1(inputs)
        ...
        return outputs
```

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn
- pandas (optional)

Install requirements using:

```bash
pip install -r requirements.txt
```

## 📊 Evaluation

Each model is trained and evaluated using accuracy, loss curves, and confusion matrices to compare performance across architectures.

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/malaria-cnn.git
cd malaria-cnn
```

2. Train any model by running its respective script:

```bash
python train_sequential.py
python train_functional.py
python train_subclassing.py
```

## 📈 Results

- All three models achieve strong classification performance.
- The Functional and Subclassing models allow more flexibility for advanced experiments.
- Model performance varies slightly depending on architecture depth and regularization.

## 📌 License

This project is open source and available under the [MIT License](LICENSE).

---

> Made with ❤️ using TensorFlow and Keras