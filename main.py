
from Data_preparation.data_loader import load_and_prepare_for_svm
from Models.svm_model import train_svm, evaluate_svm, MODEL_FILENAME, save_model, load_model
from Data_preparation.data_loader import load_and_prepare_for_cnn
from Models.cnn_model import build_cnn_model, train_cnn, evaluate_cnn, save_cnn_model

def run_svm_project():
    print("==============================================")
    print("       BẮT ĐẦU DỰ ÁN PHÂN LOẠI MNIST (SVM)     ")
    print("==============================================")
    
    # 1. Tải và chuẩn bị dữ liệu cho SVM (Đã làm phẳng, nhãn số nguyên)
    X_train, y_train, X_test, y_test = load_and_prepare_for_svm()
    
    # 2. Huấn luyện mô hình
    MAX_SAMPLES_FOR_SVM = None 
    
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

def run_cnn_project():
    print("==============================================")
    print("     BẮT ĐẦU DỰ ÁN PHÂN LOẠI MNIST (CNN)      ")
    print("==============================================")
    
    # 1. Chuẩn bị dữ liệu
    X_train, y_train, X_test, y_test = load_and_prepare_for_cnn()
    
    # 2. Xây dựng và huấn luyện
    model = build_cnn_model()
    model, history = train_cnn(model, X_train, y_train, epochs=5) # 5 lần forward và backpropagation
    
    # 3. Lưu và đánh giá
    save_cnn_model(model, 'Results/mnist_cnn_model.h5')
    final_accuracy = evaluate_cnn(model, X_test, y_test)
    
    print(f"\n--- TÓM TẮT CNN ---")
    print(f"Độ chính xác cuối cùng: {final_accuracy * 100:.4f}%")
    
    return final_accuracy

if __name__ == '__main__':
    # Chạy mô hình SVM
    # run_svm_project()

    # Chạy mô hình CNN
    run_cnn_project()