# MNIST Handwritten Digit Classification

A comprehensive project comparing Convolutional Neural Networks (CNN) and Support Vector Machines (SVM) for handwritten digit recognition using the MNIST dataset.

## 📊 Project Overview

This project implements and compares two different approaches for classifying handwritten digits:
- **Deep Learning**: CNN achieving 99.05% test accuracy
- **Traditional ML**: SVM achieving 97.92% test accuracy

The project includes data preprocessing, model training, evaluation with confusion matrices, and a GUI demo application.

## ✨ Features

- **CNN Architecture**: 3 convolutional layers with max pooling
- **SVM Implementation**: RBF kernel with optimized parameters
- **Comprehensive Evaluation**: Confusion matrices and classification reports
- **Data Preprocessing**: Normalization and appropriate formatting for each model
- **Interactive Demo**: GUI application for testing models on random MNIST samples
- **Modular Code Structure**: Separate modules for data loading, model building, and evaluation

## 🏗️ Project Structure

```
├── main.py                 # Main script to run training and evaluation
├── Data_preparation/
│   └── data_loader.py      # Data loading and preprocessing functions
├── Models/
│   ├── cnn_model.py        # CNN architecture and training functions
│   └── svm_model.py        # SVM model and training functions
├── Results/
│   ├── mnist_cnn_model.h5  # Trained CNN model
│   ├── svm_model.joblib    # Trained SVM model
│   ├── Result CNN.txt      # CNN training and evaluation results
│   └── Result SVM.txt      # SVM training and evaluation results
├── Test/
│   └── app.py              # GUI demo application
└── Report/
    └── ...                 # LaTeX report files
```

## 📋 Requirements

- Python 3.7+
- TensorFlow 2.x
- scikit-learn
- NumPy
- Pandas
- Joblib
- Pillow (PIL)
- Tkinter (usually included with Python)

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Digit-classification-MNIST.git
   cd Digit-classification-MNIST
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install tensorflow scikit-learn numpy pandas joblib pillow
   ```

## 💻 Usage

### Training Models

Run the main script to train both models:

```bash
python main.py
```

This will:
- Load and preprocess MNIST data
- Train the CNN model (5 epochs)
- Train the SVM model
- Evaluate both models on test set
- Save trained models to `Results/` directory
- Display confusion matrices and classification reports

### Running the Demo

Launch the interactive GUI demo:

```bash
python Test/app.py
```

The demo allows you to:
- View random MNIST test images
- Filter images by digit label
- Test both CNN and SVM predictions
- Compare model performance visually

## 📈 Results

### CNN Performance
- **Test Accuracy**: 99.05%
- **Architecture**: 3 Conv2D layers + 2 Dense layers
- **Training Time**: ~47 seconds (5 epochs)
- **Key Strengths**: Excellent performance on all digits, minimal misclassifications

### SVM Performance
- **Test Accuracy**: 97.92%
- **Kernel**: RBF
- **Training Time**: ~195 seconds
- **Key Characteristics**: Good baseline performance, slightly more misclassifications than CNN

### Confusion Matrix Analysis

Both models show strong performance across all digit classes. The CNN demonstrates superior accuracy, particularly in distinguishing visually similar digits like 4/9 and 3/8.

## 🔧 Model Details

### CNN Architecture
```python
Conv2D(32, (3,3), relu) -> MaxPool2D(2,2)
Conv2D(64, (3,3), relu) -> MaxPool2D(2,2)
Conv2D(64, (3,3), relu)
Flatten -> Dense(64, relu) -> Dense(10, softmax)
```

### SVM Configuration
- **Kernel**: RBF
- **C**: 1.0
- **Gamma**: scale
- **Input**: Flattened 784-dimensional vectors

## 📝 Data Preprocessing

- **Normalization**: Pixel values scaled to [0,1]
- **CNN**: Reshaped to (28, 28, 1), one-hot encoded labels
- **SVM**: Flattened to 784-dimensional vectors, integer labels

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please check individual library licenses for commercial use.

## 📚 References

- MNIST Dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- TensorFlow Documentation
- scikit-learn Documentation

---

**Note**: This project was developed as part of a computer science curriculum to demonstrate practical machine learning implementation and comparative analysis.