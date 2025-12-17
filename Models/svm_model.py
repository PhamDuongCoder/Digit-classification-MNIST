# models/svm_model.py

import time
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

MODEL_FILENAME = 'results/svm_model.joblib'

def train_svm(X_train, y_train, max_samples=None):
    """
    Huấn luyện mô hình Support Vector Machine (SVM) với kernel RBF.
    
    Args:
        X_train (np.array): Dữ liệu huấn luyện đã được làm phẳng (Flattened).
        y_train (np.array): Nhãn huấn luyện (dạng số nguyên).
        max_samples (int, optional): Số lượng mẫu tối đa để sử dụng cho huấn luyện.
                                     Dùng để tăng tốc độ thử nghiệm ban đầu. Mặc định là None (dùng hết).
    
    Returns:
        sklearn.svm.SVC: Mô hình SVM đã được huấn luyện.
    """
    
    X_train_data = X_train
    y_train_data = y_train
    
    if max_samples is not None and max_samples < len(X_train):
        print(f"⚠️ Chỉ sử dụng {max_samples} mẫu đầu tiên để thử nghiệm nhanh.")
        X_train_data = X_train[:max_samples]
        y_train_data = y_train[:max_samples]
        
    print(f"Bắt đầu huấn luyện Support Vector Machine với {len(X_train_data)} mẫu...")
    
    start_time = time.time()
    
    # Thiết lập mô hình SVM:
    # kernel='rbf' (Radial Basis Function) thường là lựa chọn tốt nhất cho dữ liệu ảnh.
    # C=1.0 và gamma='scale' là các giá trị mặc định/khởi tạo tốt.
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    
    # Huấn luyện mô hình
    svm_model.fit(X_train_data, y_train_data)
    
    end_time = time.time()
    print(f"✅ Huấn luyện SVM hoàn tất trong {end_time - start_time:.2f} giây.")
    
    return svm_model

def save_model(model, filename):
    """Lưu mô hình đã huấn luyện vào đĩa."""
    joblib.dump(model, filename)
    print(f"✅ Mô hình SVM đã được lưu tại: {filename}")

def load_model(filename):
    """Tải mô hình đã lưu từ đĩa."""
    model = joblib.load(filename)
    print(f"✅ Mô hình SVM đã được tải từ: {filename}")
    return model

def evaluate_svm(model, X_test, y_test):
    """
    Đánh giá mô hình SVM trên tập kiểm tra.
    
    Args:
        model (sklearn.svm.SVC): Mô hình SVM đã được huấn luyện.
        X_test (np.array): Dữ liệu kiểm tra đã được làm phẳng.
        y_test (np.array): Nhãn kiểm tra (dạng số nguyên).
        
    Returns:
        float: Độ chính xác của mô hình.
    """
    print("Bắt đầu đánh giá mô hình SVM...")
    
    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)
    
    # Tính toán độ chính xác
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✨ Độ chính xác của SVM: {accuracy * 100:.4f}%")
    
    # In Ma trận Nhầm lẫn và Báo cáo Phân loại
    print("\n--- Ma trận Nhầm lẫn (Confusion Matrix) ---")
    cm = confusion_matrix(y_test, y_pred)
    # Hiển thị Ma trận dưới dạng DataFrame để dễ đọc hơn
    cm_df = pd.DataFrame(cm, index=[i for i in range(10)], columns=[i for i in range(10)])
    print(cm_df)
    
    print("\n--- Báo cáo Phân loại (Classification Report) ---")
    report = classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)])
    print(report)
    
    return accuracy