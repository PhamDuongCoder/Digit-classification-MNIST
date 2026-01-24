import tkinter as tk
from tkinter import ttk
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from PIL import Image, ImageTk
import os

class MNISTDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Classifier Demo - SoICT Project")
        self.root.geometry("800x600")

        # --- 1. Tải dữ liệu và Model ---
        (_, _), (self.X_test, self.y_test) = mnist.load_data()
        
        # Sử dụng đường dẫn tuyệt đối để tránh lỗi
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        svm_path = os.path.join(project_dir, 'Results', 'svm_model.joblib')
        cnn_path = os.path.join(project_dir, 'Results', 'mnist_cnn_model.h5')
        
        self.svm_model = joblib.load(svm_path)
        self.cnn_model = load_model(cnn_path)
        self.current_img = None
        self.current_label = None

        self.create_widgets()

    def create_widgets(self):
        # --- Khung chọn nhãn (Top) ---
        top_frame = tk.Frame(self.root, pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(top_frame, text="Chọn nhãn mục tiêu:").pack(side=tk.LEFT, padx=10)
        self.label_var = tk.StringVar(value="None")
        options = ["None"] + [str(i) for i in range(10)]
        self.dropdown = ttk.Combobox(top_frame, textvariable=self.label_var, values=options, state="readonly", width=10)
        self.dropdown.pack(side=tk.LEFT, padx=5)

        btn_random = tk.Button(top_frame, text="Hiển thị random ảnh", command=self.load_random_image, bg="#e1e1e1")
        btn_random.pack(side=tk.LEFT, padx=20)

        # --- Khung chính (Main Layout) ---
        main_frame = tk.Frame(self.root)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        # Cột trái: Hiển thị ảnh
        self.left_col = tk.Frame(main_frame, bd=2, relief=tk.SUNKEN, width=400, height=400)
        self.left_col.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10)
        self.left_col.pack_propagate(False)

        self.img_display = tk.Label(self.left_col, text="Ảnh sẽ hiển thị ở đây")
        self.img_display.pack(expand=True)

        self.btn_reset = tk.Button(self.left_col, text="Reset", command=self.reset_all, width=15)
        self.btn_reset.pack(pady=20)

        # Cột phải: Dự đoán
        right_col = tk.Frame(main_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        # Phần SVM
        tk.Button(right_col, text="Mô hình SVM dự đoán", command=self.predict_svm, height=2, width=25).pack(pady=5)
        self.svm_res_box = tk.Label(right_col, text="Kết quả dự đoán\nmô hình SVM", bd=2, relief=tk.GROOVE, width=30, height=8, bg="white")
        self.svm_res_box.pack(pady=5)

        # Khoảng cách
        tk.Label(right_col, text="").pack(pady=10)

        # Phần CNN
        tk.Button(right_col, text="Mô hình CNN dự đoán", command=self.predict_cnn, height=2, width=25).pack(pady=5)
        self.cnn_res_box = tk.Label(right_col, text="Kết quả dự đoán\nmô hình CNN", bd=2, relief=tk.GROOVE, width=30, height=8, bg="white")
        self.cnn_res_box.pack(pady=5)

    def load_random_image(self):
        target = self.label_var.get()
        if target == "None":
            idx = np.random.randint(0, len(self.X_test))
        else:
            indices = np.where(self.y_test == int(target))[0]
            idx = np.random.choice(indices)
        
        self.current_img = self.X_test[idx]
        self.current_label = self.y_test[idx]
        
        # Chuyển mảng numpy sang ảnh hiển thị
        img = Image.fromarray(self.current_img).resize((300, 300))
        img_tk = ImageTk.PhotoImage(image=img)
        self.img_display.config(image=img_tk, text="")
        self.img_display.image = img_tk

    def predict_svm(self):
        if self.current_img is None: return
        # Tiền xử lý: Flatten & Normalize
        X = self.current_img.reshape(1, -1) / 255.0
        pred = self.svm_model.predict(X)[0]
        color = "green" if pred == self.current_label else "red"
        self.svm_res_box.config(text=f"Dự đoán: {pred}\nThực tế: {self.current_label}", fg=color, font=("Arial", 12, "bold"))

    def predict_cnn(self):
        if self.current_img is None: return
        # Tiền xử lý: Reshape 4D & Normalize
        X = self.current_img.reshape(1, 28, 28, 1) / 255.0
        probs = self.cnn_model.predict(X, verbose=0)
        pred = np.argmax(probs)
        color = "green" if pred == self.current_label else "red"
        self.cnn_res_box.config(text=f"Dự đoán: {pred}\nThực tế: {self.current_label}", fg=color, font=("Arial", 12, "bold"))

    def reset_all(self):
        self.current_img = None
        self.img_display.config(image="", text="Ảnh sẽ hiển thị ở đây")
        self.svm_res_box.config(text="Kết quả dự đoán\nmô hình SVM", fg="black", font=("Arial", 10))
        self.cnn_res_box.config(text="Kết quả dự đoán\nmô hình CNN", fg="black", font=("Arial", 10))

if __name__ == "__main__":
    root = tk.Tk()
    app = MNISTDemoApp(root)
    root.mainloop()