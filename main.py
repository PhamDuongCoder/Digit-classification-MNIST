# main.py

# Giả định: data_loader.py của bạn có hàm load_and_prepare_for_svm
# Hàm này phải trả về (X_train_svm, y_train_svm, X_test_svm, y_test_svm)
from Data_preparation.data_loader import load_and_prepare_for_svm
from Models.svm_model import train_svm, evaluate_svm, MODEL_FILENAME, save_model, load_model

def run_svm_project():
    print("==============================================")
    print("       BẮT ĐẦU DỰ ÁN PHÂN LOẠI MNIST (SVM)     ")
    print("==============================================")
    
    # 1. Tải và chuẩn bị dữ liệu cho SVM (Đã làm phẳng, nhãn số nguyên)
    X_train, y_train, X_test, y_test = load_and_prepare_for_svm()
    
    # 2. Huấn luyện mô hình
    # 💡 Lời khuyên: Đặt max_samples=10000 để chạy nhanh khi thử nghiệm
    # Khi chạy chính thức, hãy đặt max_samples=None để dùng toàn bộ 60,000 mẫu
    MAX_SAMPLES_FOR_SVM = None # Hoặc None để dùng hết
    
    svm_model = train_svm(
        X_train=X_train, 
        y_train=y_train, 
        max_samples=MAX_SAMPLES_FOR_SVM
    )

    # Lưu mô hình đã huấn luyện
    save_model(svm_model, MODEL_FILENAME)
    
    # 3. Đánh giá mô hình
    print("\n--- Kết quả Đánh giá trên tập Kiểm tra ---")
    final_accuracy = evaluate_svm(svm_model, X_test, y_test)
    
    print(f"\n--- TÓM TẮT SVM ---")
    print(f"Độ chính xác cuối cùng: {final_accuracy * 100:.4f}%")
    
    return final_accuracy

if __name__ == '__main__':
    # Chạy mô hình SVM
    run_svm_project()
    
    # Bạn có thể thêm hàm run_cnn_project() vào đây sau khi hoàn thành Giai đoạn 3
    # print("\n" + "="*50 + "\n")
    # run_cnn_project()