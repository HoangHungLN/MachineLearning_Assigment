# Bài Tập Lớn Học Máy – CO3117 (Nhóm CEML2, Lớp TN01)

## Thông tin môn học
- **Tên môn học:** Học Máy  
- **Mã môn:** CO3117  
- **Lớp:** TN01 – Nhóm CEML2  
- **Học kỳ:** 251, Năm học 2025 – 2026  

## Giảng viên hướng dẫn
- **TS. Lê Thành Sách**

## Thành viên nhóm
- **Trương Thiên Ân** – 2310190 – an.truong241105@hcmut.edu.vn  
- **Lại Nguyễn Hoàng Hưng** – 2311327 – hung.lai2805@hcmut.edu.vn  
- **Nguyễn Tô Quốc Việt** – 2313898 – viet.nguyenluminous@hcmut.edu.vn  

---

## Mục tiêu bài tập lớn
 1. Hiểu và áp dụng được quy trình pipeline học máy truyền thống, bao gồm: tiền xử lý dữ liệu, trích xuất đặc trưng, huấn luyện và đánh giá mô hình.
 2. Rèn luyện kỹ năng triển khai mô hình học máy trên các loại dữ liệu khác nhau: bảng, văn bản, và ảnh.
 3. Phát triển khả năng phân tích, so sánh, và đánh giá hiệu quả của các mô hình học máy thông qua các chỉ số đo lường.
 4. Rèn luyện kỹ năng lập trình, thử nghiệm, và tổ chức báo cáo khoa học
 
## Assignment 1
### Mục tiêu bài tập
1. **Xử lý dữ liệu đầu vào**  
   - Thực hành xử lý giá trị thiếu (*missing values*) bằng kỹ thuật imputation.  
   - Thực hành mã hóa biến phân loại (*categorical features*) bằng kỹ thuật encoding.  

2. **Xây dựng pipeline học máy cho dữ liệu dạng bảng (Tabular Data)**  
   - Chuẩn hóa dữ liệu bằng các kỹ thuật impute và encoding.  
   - Lựa chọn và thực hiện giảm chiều dữ liệu bằng PCA (nếu cần).  
   - Áp dụng các mô hình học máy (ví dụ: Logistic Regression, SVM, Random Forest).  

3. **So sánh và đánh giá mô hình**  
   - So sánh hiệu quả giữa các mô hình đã huấn luyện.  
   - Đưa ra báo cáo kết quả: phân tích dữ liệu (EDA), mô tả pipeline, cấu hình các bước xử lý, và đánh giá.  
---

### Dataset
- **Tên:** *Mobile Phones in Indian Market Datasets*  
- **Nguồn:** [Kaggle Link](https://www.kaggle.com/datasets/kiiroisenkoxx/2025-mobile-phones-in-indian-market-datasets/data?select=mobiles_uncleaned.csv)  
- **Mô tả:** 11.786 mẫu, 14 thuộc tính về đặc điểm kỹ thuật và thông tin của các dòng điện thoại.  
- **Mục tiêu:** phân loại điện thoại theo giá (`low / medium / high`).  

**Cách tải dataset trong Colab:**  
Dataset đã được push lên GitHub, đã được cấu hình sẵn trong notebook để đảm bảo sẽ tự động tải sau khi nhấn Run Time -> Run all
### Mô tả các module
- **`__init__.py`**:  
  Khai báo và gom tất cả hàm trong `feature_extractors.py` để tiện import (`extract_is_dual_sim`, `extract_cpu_speed`, `extract_ram`, ...).  

- **`feature_extractors.py`**:  
  Chứa các hàm *feature engineering* để trích xuất đặc trưng từ dữ liệu thô (chuỗi văn bản) thành dạng số:  
  - `extract_is_dual_sim`, `extract_is_5g`, `extract_is_nfc`  
  - `extract_cpu_brand`, `extract_cpu_speed`, `extract_cpu_core`  
  - `extract_ram`, `extract_rom`, `extract_battery`, `extract_fast_charging`  
  - `extract_screen_size`, `extract_refresh_rate`, `extract_ppi`  
  - `extract_rear`, `extract_front_camera`  
  - `extract_expandable_storage`, `extract_os`  

- **`model_runner.py`**:  
  Định nghĩa hàm `run_model(...)` để xây dựng pipeline:  
  - Tiền xử lý dữ liệu (imputation, scaling, encoding).  
  - Giảm chiều dữ liệu bằng PCA.  
  - Huấn luyện mô hình (Logistic Regression, SVM, Random Forest).  
  - Trả về metrics (Accuracy, Precision, Recall, F1, Explained Variance %).

-----
## Assignment 2 
### Mục tiêu bài tập
1. **Xử lý dữ liệu đầu vào**  
   - Làm sạch văn bản: loại bỏ ký tự đặc biệt, chuẩn hóa chữ thường.
   - Thực hiện tokenization, loại bỏ stopwords, và padding độ dài chuỗi (nếu cần).
   - Xây dựng lớp TextPreprocessor linh hoạt, cho phép bật/tắt từng bước tiền xử lý thông qua tham số.
2. **Xây dựng pipeline học máy cho dữ liệu dạng bảng (Tabular Data)**  
   - Trích xuất đặc trưng bằng các phương pháp: Bag-of-Words (BoW), TF–IDF (Term Frequency–Inverse Document Frequency), TF–IDF Weighted GloVe Embedding
   - Thiết kế pipeline cho phép cấu hình mô hình và đặc trưng linh hoạt (BoW, TF–IDF, GloVe).
   - Huấn luyện các mô hình học máy: Naive Bayes, Logistic Regression, SVM (LinearSVC).
3. **So sánh và đánh giá mô hình**  
   - Thử nghiệm và tinh chỉnh tham số (Hyperparameter Tuning):
   - Đánh giá mô hình trên tập validation và test bằng các chỉ số: Accuracy, Precision, Recall, F1-score.
   - So sánh hiệu quả giữa các phương pháp trích xuất đặc trưng (BoW, TF–IDF, GloVe) và mô hình.
   - Phân tích so sánh giữa cách tiếp cận truyền thống (BoW, TF–IDF) và hiện đại (TF–IDF Weighted GloVe).
---

### Dataset
- **Tên:** *"IT Service Ticket Classification Dataset"*  
- **Nguồn:** [Kaggle Link](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset)  
- **Mô tả:** 47,837  mẫu, 8 chủ đề phân loại.  
- **Mục tiêu:** Phân loại chủ đề của các đoạn yêu cầu dịch vụ. 

**Cách tải dataset trong Colab:**  
Dataset đã được push lên GitHub, đã được cấu hình sẵn trong notebook để đảm bảo sẽ tự động tải sau khi nhấn Run Time -> Run all
### Mô tả các module

- **`features_extractor.py`**:  
  Chứa các hàm *tiền xử lý* và *trích xuất đặc trưng cơ bản* cho dữ liệu văn bản.  
  - Làm sạch dữ liệu: loại bỏ ký tự đặc biệt, chuyển về chữ thường, tách từ, v.v.  
  - Tạo và lưu các đặc trưng văn bản bằng các phương pháp phổ biến như:
    - **Bag of Words (BoW)**
    - **TF-IDF (Term Frequency – Inverse Document Frequency)**
    - **TF-IDF + GloVe** (kết hợp biểu diễn thống kê và ngữ nghĩa).  
  - Các hàm tiêu biểu:
    - `build_bow_features()` – sinh đặc trưng BoW.  
    - `build_tfidf_features()` – sinh đặc trưng TF-IDF.  
    - `clean_text()` – tiền xử lý văn bản đầu vào.  

- **`tfidf_glove.py`**:  
  Cài đặt quy trình kết hợp **TF-IDF weighting** với **GloVe embeddings** để biểu diễn văn bản ở dạng vector dense.  
  - `load_glove_model()` – tải và chuyển đổi file GloVe sang định dạng Word2Vec.  
  - `build_tfidf()` – huấn luyện TF-IDF để sinh bản đồ IDF cho từng từ.  
  - `sent_vec_tfidf()` – tính vector câu dựa trên trung bình có trọng số TF-IDF của các từ.  
  - `docs_to_matrix()` – chuyển toàn bộ tập văn bản thành ma trận đặc trưng.  
  - `run_tfidf_glove()` – pipeline chính:
    - Tokenize văn bản.  
    - Load mô hình GloVe.  
    - Áp dụng TF-IDF weighting.  
    - Sinh và lưu các ma trận đặc trưng `Xtr_w2v.npy`, `Xva_w2v.npy`, `Xte_w2v.npy`.  

- **`models.py`**:  
  Định nghĩa các hàm huấn luyện và đánh giá mô hình học máy cổ điển: **Naive Bayes**, **Logistic Regression**, và **SVM**.  
  - `run_models(Xtr, ytr, Xva, yva, Xte, yte, model_params)`  
    - Huấn luyện các mô hình dựa trên đặc trưng đầu vào.  
    - Thử nghiệm các bộ **hyperparameters** khác nhau.  
    - Trả về độ chính xác (*validation accuracy*) của từng mô hình.  
  - `evaluate_model_on_test(model, Xte, yte, model_name)`  
    - Đánh giá mô hình tốt nhất trên tập test.  
    - In ra **classification report** gồm *precision*, *recall*, *f1-score* cho từng lớp.  
 
-----
## Assignment 3 (comming soon)
-----
## Phần mở rộng (comming soon)
---

##  Hướng dẫn chạy notebook
- Mở notebook **`Assignment1_CEML2.ipynb`** trong Google Colab.  
- Chọn **Runtime → Run All**.  
- Notebook đã được cấu hình sẵn: import thư viện, tải dataset, xử lý và chạy mô hình.  
- Sau khi chạy, bạn sẽ có ngay kết quả huấn luyện và đánh giá.  

---

## Cấu trúc dự án
```
MachineLearning_Assignment/
├── Assignment1/
│   ├── data/
│   │   └── mobiles_uncleaned.csv
│   ├── modules/
│   │   ├── features_extractor.py
│   │   ├── model_runner.py
│   │   └── __init__.py
│   └── notebooks/
│       └── Assignment1_CEML2.ipynb
│
Assignment2/
├── data/
│   └── all_tickets_processed_improved_v3.csv
│
├── features/
│   └── tfidf_glove/
│       ├── Xte_w2v.npy
│       ├── Xtr_w2v.npy
│       └── Xva_w2v.npy
│
├── modules/
│   ├── features_extractor.py
│   ├── models.py
│   └── tfidf_glove.py
│
├── notebooks/
│   └── Assignment2_CEML2.ipynb
│
|
├── Assignment3/
│   ├── data/
│   ├── modules/
│   └── notebooks/
│
└── README.md

```

## Notebook
 
- [Link notebook Assignment 1](https://colab.research.google.com/drive/1saG65yL3ieFIaZLorNRLfMgdfchSFudX?usp=sharing)
- [Link notebook Assignment 2](https://colab.research.google.com/github/HoangHungLN/MachineLearning_Assigment/blob/main/Assignment2/notebooks/Assignment2_CEML2.ipynb)

---

## Liên hệ
Nếu có thắc mắc, vui lòng liên hệ:  
- **Trương Thiên Ân** – an.truong241105@hcmut.edu.vn  
- **Lại Nguyễn Hoàng Hưng** – hung.lai2805@hcmut.edu.vn  
- **Nguyễn Tô Quốc Việt** – viet.nguyenluminous@hcmut.edu.vn  

## Acknowledgement
Đây là bản sao bài tập lớn môn **CO3117 – Học Máy** (Lớp TN01, Nhóm CEML2, Học kỳ 251, Năm học 2025–2026).  
