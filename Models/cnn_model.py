# models/cnn_model.py

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import time
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

class TimingCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        steps = self.params['steps']
        ms_per_step = (epoch_time / steps) * 1000
        print(f"Epoch {epoch+1}/{self.params['epochs']}: {steps}/{steps}/tensorflow - {epoch_time:.2f}s - {ms_per_step:.0f}ms/step - Accuracy: {logs['accuracy']:.4f} - Loss: {logs['loss']:.4f} - Val_accuracy: {logs['val_accuracy']:.4f} - Val_loss: {logs['val_loss']:.4f}")

def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """Xây dựng cấu trúc mạng CNN."""
    model = models.Sequential([
        # Lớp Tích chập 1: Trích xuất các nét cơ bản 
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Lớp Tích chập 2: Trích xuất các đặc trưng phức tạp hơn
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Lớp Tích chập 3
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Phẳng hóa để đưa vào các lớp fully connected
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        
        # Lớp đầu ra với Softmax để phân loại 10 chữ số
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_cnn(model, X_train, y_train, epochs=10, batch_size=64):
    """Huấn luyện mô hình CNN."""
    print(f"Bắt đầu huấn luyện CNN trong {epochs} epochs...")
    start_time = time.time()
    
    # Huấn luyện
    history = model.fit(X_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_split=0.1, # Dùng 10% data train để validate
                        verbose=0,
                        callbacks=[TimingCallback()])
    
    end_time = time.time()
    print(f"Huấn luyện CNN hoàn tất trong {end_time - start_time:.2f} giây.")
    return model, history

def save_cnn_model(model, filename='results/cnn_model.h5'):
    """Lưu mô hình CNN"""
    model.save(filename)
    print(f"Mô hình CNN đã được lưu tại: {filename}")

def evaluate_cnn(model, X_test, y_test):
    """Đánh giá mô hình CNN trên tập test."""
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nĐộ chính xác của CNN trên tập kiểm tra: {test_acc * 100:.4f}%")
    
    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    
    # In ma trận nhầm lẫn và báo cáo phân loại
    print("\n--- Ma trận Nhầm lẫn (Confusion Matrix) ---")
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    # Hiển thị ma trận dưới dạng DataFrame để dễ đọc hơn
    cm_df = pd.DataFrame(cm, index=[i for i in range(10)], columns=[i for i in range(10)])
    print(cm_df)
    
    print("\n--- Báo cáo Phân loại (Classification Report) ---")
    report = classification_report(y_test_classes, y_pred_classes, target_names=[str(i) for i in range(10)])
    print(report)
    
    return test_acc