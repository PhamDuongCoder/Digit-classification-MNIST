# data_preparation/data_loader.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def load_and_prepare_for_svm():
    """
    Tải và tiền xử lý dữ liệu MNIST sẵn sàng cho mô hình SVM.
    
    Các bước tiền xử lý:
    1. Chuẩn hóa pixel về phạm vi [0.0, 1.0].
    2. Làm phẳng (Flatten) ảnh 28x28 thành vector 784.
    3. Giữ nguyên nhãn dưới dạng số nguyên (0-9).
    
    Returns:
        tuple: (X_train_svm, y_train_svm, X_test_svm, y_test_svm)
    """
    
    print("--- Bắt đầu tải và tiền xử lý dữ liệu cho SVM ---")
    
    # 1. Tải dữ liệu
    # Dữ liệu được tải dưới dạng (60000, 28, 28) và nhãn (60000,)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # --- Tiền xử lý Dữ liệu X (Features) ---
    
    # 2. Chuẩn hóa (Normalization)
    # Chuyển đổi sang float32 và chia cho 255.0
    X_train_norm = X_train.astype('float32') / 255.0
    X_test_norm = X_test.astype('float32') / 255.0
    
    # 3. Làm phẳng (Flatten) cho SVM
    # Kích thước mới: (số mẫu, 784)
    num_pixels = X_train_norm.shape[1] * X_train_norm.shape[2] # 28 * 28 = 784
    
    X_train_svm = X_train_norm.reshape(X_train_norm.shape[0], num_pixels)
    X_test_svm = X_test_norm.reshape(X_test_norm.shape[0], num_pixels)
    
    # --- Tiền xử lý Nhãn y (Labels) ---
    
    # 4. Nhãn cho SVM: Giữ nguyên dạng số nguyên (0-9)
    y_train_svm = y_train
    y_test_svm = y_test
    
    print(f"Kích thước X_train_svm sau làm phẳng: {X_train_svm.shape}")
    print(f"Kích thước y_train_svm: {y_train_svm.shape}")
    print("--- Hoàn tất chuẩn bị dữ liệu cho SVM ---")
    
    return X_train_svm, y_train_svm, X_test_svm, y_test_svm


# Data loader cho model cnn
def load_and_prepare_for_cnn():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # 1. Reshape về (28, 28, 1) và chuẩn hóa
    X_train_cnn = X_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    X_test_cnn = X_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    
    # 2. One-hot encoding cho nhãn
    y_train_cnn = to_categorical(y_train, 10)
    y_test_cnn = to_categorical(y_test, 10)
    
    return X_train_cnn, y_train_cnn, X_test_cnn, y_test_cnn