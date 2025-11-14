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
## Assignment 3
### Mục tiêu bài tập

  1. Xử lý dữ liệu ảnh đầu vào
     * Đọc danh sách ảnh và nhãn từ file CSV (`Training_set.csv`).
     * Chuẩn hóa ảnh: resize về kích thước cố định (224×224), chuẩn hóa giá trị pixel bằng hàm `preprocess_input` tương ứng với từng mô hình pretrained.
     * Chia dữ liệu thành các tập `train / val / test` theo tỷ lệ cấu hình, đảm bảo phân bố nhãn cân bằng (stratified split).

  2. Trích xuất đặc trưng từ mô hình CNN pretrained
     * Sử dụng các mô hình đã được huấn luyện trước trên ImageNet: **VGG16**, **ResNet50**, **EfficientNetB0** làm backbone.
     * Đóng băng trọng số (freeze) backbone và thêm tầng `GlobalAveragePooling2D` để trích xuất vector đặc trưng cho từng ảnh.
     * Lưu lại đặc trưng và nhãn tương ứng dưới dạng file `.npy` để tái sử dụng cho nhiều mô hình học máy khác nhau mà không cần trích xuất lại.

  3. Xây dựng pipeline học máy cho dữ liệu ảnh
     * Tận dụng đặc trưng từ CNN (feature vector) để huấn luyện các mô hình học máy cổ điển: **Logistic Regression**, **LinearSVC**, **SVC**, **RandomForest**.
     * Hỗ trợ các bước tiền xử lý tuỳ chọn:
       * Chuẩn hóa (Normalization/Standardization).
       * Giảm chiều dữ liệu bằng **PCA** (giữ lại ~95% phương sai).
     * Tổ chức pipeline linh hoạt để dễ dàng thử nghiệm các combination:
       * Backbone (VGG16 / ResNet50 / EfficientNetB0).
       * Mô hình phân loại (LR / SVM / RF).
       * Có / không dùng PCA, chuẩn hóa.

  4. So sánh và đánh giá mô hình
     * Đo lường chất lượng mô hình trên tập validation và test với các chỉ số:
       * **Accuracy**, **Precision**, **Recall**, **F1-score**.
     * So sánh:
       * Giữa các backbone CNN khác nhau.
       * Giữa các mô hình phân loại cổ điển.
       * Ảnh hưởng của chuẩn hóa và PCA lên hiệu năng.
     * Chọn ra cấu hình mô hình tối ưu và phân tích kết quả bằng báo cáo phân loại (classification report).
    
### Dataset

  * Tên: **Human Action Recognition (HAR) – 15 Human Activities**
  * Nguồn: [Kaggle Link](https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset)
  * Mô tả:
    * ~12.600 ảnh huấn luyện và ~5.400 ảnh test.  
    * Mỗi ảnh thuộc đúng **1 trong 15 lớp hành động**:
      `calling, clapping, cycling, dancing, drinking, eating, fighting, hugging, laughing, listening_to_music, running, sitting, sleeping, texting, using_laptop`.  
    * Ảnh được chụp trong nhiều bối cảnh khác nhau, độ đa dạng cao về góc chụp, môi trường và đối tượng, phù hợp cho bài toán phân loại hành động người từ ảnh tĩnh. :contentReference[oaicite:0]{index=0}
  * Mục tiêu: xây dựng hệ thống nhận dạng hành động con người từ ảnh tĩnh dựa trên đặc trưng trích xuất từ các mô hình CNN pretrained kết hợp với các mô hình học máy cổ điển.

Cách tải dataset trong Colab:  
Dataset (bao gồm ảnh và file `Training_set.csv`) đã được cấu hình sẵn trong notebook để đảm bảo sẽ tự động tải / đọc từ thư mục `Assignment3/data/` sau khi nhấn **Runtime → Run all**.

* * *
### Mô tả các module

  * `preprocessor.py`  
    Chứa hàm `preprocessing(...)` chịu trách nhiệm:
    * Đọc file CSV chứa tên file và nhãn (`filename`, `label`).
    * Ghép đường dẫn đầy đủ đến ảnh từ thư mục `image_dir`.
    * Chia dữ liệu thành `train / val / test` với stratified split.
    * Áp dụng hàm `preprocess_input` tương ứng với từng mô hình (**VGG16**, **ResNet50**, **EfficientNetB0**) để chuẩn hóa ảnh.
    * Tạo các `tf.data.Dataset` cho train, val, test với batching và prefetch để tăng tốc huấn luyện. :contentReference[oaicite:1]{index=1}

  * `feature_extractor.py`  
    Định nghĩa hàm `run_extraction(model_name, data_pipeline, output_dir)`:
    * Khởi tạo backbone CNN tương ứng với `model_name`:
      * `resnet50` → `tf.keras.applications.ResNet50`
      * `vgg16` → `tf.keras.applications.VGG16`
      * `efficientnetb0` → `tf.keras.applications.EfficientNetB0`
    * Đóng băng trọng số và gắn thêm tầng `GlobalAveragePooling2D` tạo thành feature extractor.
    * Lặp qua từng batch trong `train / val / test`:
      * Chạy forward (inference mode) để lấy vector đặc trưng.
      * Gom các batch lại thành ma trận `X_split` và vector nhãn `y_split`.
    * Lưu kết quả ra các file:
      * `X_train.npy`, `y_train.npy`
      * `X_val.npy`, `y_val.npy`
      * `X_test.npy`, `y_test.npy` trong thư mục `output_dir`. :contentReference[oaicite:2]{index=2}

  * `run_model.py`  
    Cài đặt các hàm huấn luyện và đánh giá mô hình học máy cổ điển:
    * `run_models(Xtr, ytr, Xva, yva, Xte, yte, model_params, print_reports=True, pca=False, normalize=False)`:
      * Xây dựng và huấn luyện các mô hình:
        * **RandomForestClassifier**
        * **LogisticRegression**
        * **LinearSVC**
        * **SVC**
      * Tùy chọn:
        * Chuẩn hóa đặc trưng bằng `StandardScaler` hoặc `Normalizer`.
        * Giảm chiều bằng **PCA** với `n_components=0.95`.
      * Trả về dictionary chứa mô hình đã huấn luyện và validation accuracy cho từng phương án.
    * `evaluate_model_on_test(model, Xte, yte, model_name)`:
      * Đánh giá mô hình tốt nhất trên tập test.
      * In ra **Test Accuracy** và `classification_report` (precision, recall, f1-score cho từng lớp). :contentReference[oaicite:3]{index=3}

  * `Assignment3_CEML2.ipynb`  
    Notebook chính để:
    * Cấu hình tham số (chọn backbone CNN, mô hình phân loại, bật/tắt chuẩn hóa & PCA).
    * Gọi lần lượt:
      * `preprocessing(...)` → tạo pipeline dữ liệu ảnh.
      * `run_extraction(...)` → sinh đặc trưng và lưu `.npy`.
      * `run_models(...)` → huấn luyện nhiều mô hình trên đặc trưng đã trích xuất.
      * `evaluate_model_on_test(...)` → đánh giá mô hình tốt nhất trên tập test.
    * Tổng hợp và trực quan hóa kết quả thí nghiệm (bảng so sánh accuracy, báo cáo chi tiết theo lớp).
-----
## Phần mở rộng

### Mục tiêu bài tập

1. Ôn tập và mở rộng kiến thức về mô hình đồ thị xác suất cho bài toán chuỗi (sequence labeling).
2. Hiểu và triển khai mô hình **Hidden Markov Model (HMM)** cho bài toán **gán nhãn từ loại (POS tagging)**.
3. Tự cài đặt từ đầu hai giải thuật cơ bản trên HMM:
   - **Giải thuật Forward** – tính xác suất xuất hiện của cả chuỗi quan sát.
   - **Giải thuật Viterbi** – tìm chuỗi nhãn (tags) có xác suất cao nhất.
4. Thực hành đánh giá mô hình trên tập dữ liệu thật:
   - Tính **tagging accuracy** trên tập dev.
   - Quan sát, phân tích các lỗi điển hình (nhầm lẫn giữa các POS tag).
5. Kết nối kiến thức giữa:
   - Mô hình học máy cổ điển (Assignment 1, 2, 3),
   - Và mô hình xác suất cho dữ liệu tuần tự (Extended Assignment – HMM cho POS tagging).

* * *

### Dataset

- **Tên:** Tập dữ liệu gán nhãn từ loại tiếng Anh (POS tagging dev set).
- * Nguồn: [Kaggle Link](https://www.kaggle.com/datasets/pranav13300/annotated-dataset-for-pos-tagging?select=train.json)
- **Định dạng:** File `train.json` gồm nhiều câu, mỗi phần tử có cấu trúc:
  - `index`: chỉ số câu.
  - `sentence`: danh sách các từ trong câu.
  - `labels`: danh sách nhãn POS tương ứng với từng từ.
- **Quy mô:** khoảng 38k câu, mỗi câu là một cặp (sentence, labels).
- **Mục tiêu:**  
  Huấn luyện/đánh giá mô hình HMM cho bài toán **gán nhãn từ loại** – với mỗi từ trong câu, dự đoán nhãn POS tương ứng.

Cách tải dataset trong Colab:

- Dataset (`train.json`) đã được push sẵn trong thư mục `Extended_Assignment/`.
- Notebook `Extended_Assignment_CEML2.ipynb` đã được cấu hình đường dẫn, chỉ cần:
  - Mở notebook trên Colab.
  - Chọn **Runtime → Run all** để tự động:
    - Đọc file `dev.json`.
    - Xử lý dữ liệu đầu vào.
    - Chạy các giải thuật Forward/Viterbi và đánh giá mô hình.

* * *

### Mô tả các module

- `forward_algorithm.py`  
  Cài đặt **Giải thuật Forward** cho mô hình HMM dạng rời rạc.

  - Hàm chính: `forward_algorithm(O, pi, A, B)`  
    - `O`: chuỗi quan sát dưới dạng chỉ số (ví dụ: `[0, 5, 12, ...]`).
    - `pi`: vector xác suất trạng thái khởi đầu.
    - `A`: ma trận xác suất chuyển trạng thái, `A[i, j] = P(s_j | s_i)`.
    - `B`: ma trận xác suất phát xạ, `B[i, k] = P(o_k | s_i)`.
  - Trả về:
    - Tổng xác suất `P(O | λ)` của cả chuỗi quan sát.
  - Có xử lý ngoại lệ khi chỉ số quan sát vượt quá kích thước ma trận `B`, giúp debug dễ dàng hơn.

- `viterbi_algorithm.py`  
  Cài đặt **Giải thuật Viterbi** để tìm chuỗi nhãn POS tốt nhất cho một câu.

  - Hàm chính: `viterbi_algorithm(words, states, start_p, trans_p, emit_p)`  
    - `words`: danh sách các từ trong câu.
    - `states`: danh sách tất cả các nhãn POS (tag set).
    - `start_p`: xác suất bắt đầu cho từng tag.
    - `trans_p`: xác suất chuyển trạng thái `P(tag_t | tag_{t-1})`.
    - `emit_p`: xác suất phát xạ `P(word | tag)`.
  - Bên trong triển khai:
    - Bảng động `V[t][tag]` lưu xác suất tốt nhất tại vị trí `t` khi ở trạng thái `tag`.
    - Mảng `backpointer` để truy vết ngược lại chuỗi nhãn tốt nhất.
  - Trả về:
    - `best_tags`: chuỗi nhãn POS tối ưu.
    - `best_prob`: xác suất tương ứng của chuỗi nhãn này.

- `train.json`  
  - Chứa tập **train** đã được gán nhãn sẵn để:
    - Tính toán độ chính xác của mô hình HMM.
    - So sánh giữa chuỗi nhãn dự đoán (từ Viterbi) và nhãn ground-truth.

- `Extended_Assignment_CEML2.ipynb` (notebook – đặt trong thư mục `Extended_Assignment/notebooks/`)  
  - Kết nối toàn bộ pipeline:
    - Đọc dữ liệu từ `train.json`.
    - Xây dựng các thống kê để suy ra `start_p`, `trans_p`, `emit_p`.
    - Gọi `forward_algorithm` để tính xác suất chuỗi quan sát.
    - Gọi `viterbi_algorithm` để dự đoán chuỗi POS tags.
    - Đánh giá mô hình, in ra một số ví dụ minh hoạ (câu, nhãn thật, nhãn dự đoán).
---

##  Hướng dẫn chạy notebook
- Mở notebook **muốn chạy** trong Google Colab.  
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
    ├── Assignment2/
    │   ├── data/
    │   │   └── all_tickets_processed_improved_v3.csv
    │   ├── features/
    │   │   └── tfidf_glove/
    │   │       ├── Xte_w2v.npy
    │   │       ├── Xtr_w2v.npy
    │   │       └── Xva_w2v.npy
    │   ├── modules/
    │   │   ├── features_extractor.py
    │   │   ├── models.py
    │   │   └── tfidf_glove.py
    │   └── notebooks/
    │       └── Assignment2_CEML2.ipynb
    │
    ├── Assignment3/
    │   ├── data/
    │   │   ├── HumanActionDataset/        # folder ảnh (LFS)
    │   │   ├── Training_set.csv          # file label
    │   │   └── .gitkeep
    │   ├── features_output/
    │   │   ├── efficientnetb0/           # các file .npy trích xuất từ EfficientNetB0
    │   │   ├── resnet50/                 # các file .npy trích xuất từ ResNet50
    │   │   └── vgg16/                    # các file .npy trích xuất từ VGG16
    │   ├── modules/
    │   │   ├── __pycache__/
    │   │   ├── feature_extractor.py
    │   │   ├── preprocessor.py
    │   │   └── run_model.py
    │   └── notebooks/
    │       └── Assignment3_CEML2.ipynb
    │
    ├── Extended_Assignment/
    │   ├── data/
    │   │   ├── dev.json                  # tập dev POS tagging
    │   │   ├── train.json                # tập train POS tagging
    │   │   └── .gitkeep
    │   ├── modules/
    │   │   ├── forward_algorithm.py
    │   │   └── viterbi_algorithm.py
    │   └── notebooks/
    │       └── Extended_Assignment_CEML2.ipynb
    │
    └── README.md

```

## Notebook
 
- [Link notebook Assignment 1](https://colab.research.google.com/drive/1saG65yL3ieFIaZLorNRLfMgdfchSFudX?usp=sharing)
- [Link notebook Assignment 2](https://colab.research.google.com/github/HoangHungLN/MachineLearning_Assigment/blob/main/Assignment2/notebooks/Assignment2_CEML2.ipynb)
- [Link notebook Assignment 3](https://colab.research.google.com/github/HoangHungLN/MachineLearning_Assignment/blob/main/Assignment3/notebooks/Assignment3_CEML2.ipynb)
- [Link notebook Phần mở rộng](https://colab.research.google.com/github/HoangHungLN/MachineLearning_Assignment/blob/main/Extended_Assignment/notebooks/Extended_Assignment.ipynb)
---

## Phân chia công việc
| **Thành viên**            | **Assignment 1**                  | **Assignment 2**               | **Assignment 3**               | **Extended Assignment**                     | **Hoàn thành** |
| ------------------------- | --------------------------------- | ------------------------------ | ------------------------------ | ------------------------------------------- | -------------- |
| **Trương Thiên Ân**       | EDA, tiền xử lý dữ liệu           | Trích xuất đặc trưng           | Huấn luyện và đánh giá mô hình | Hiện thực & đánh giá giải thuật **Viterbi** | 100%           |
| **Lại Nguyễn Hoàng Hưng** | Giảm số chiều, huấn luyện mô hình | Huấn luyện và đánh giá mô hình | EDA, tiền xử lý dữ liệu        | Hiện thực & đánh giá giải thuật **Forward** | 100%           |
| **Nguyễn Tô Quốc Việt**   | Đánh giá mô hình, báo cáo         | EDA, tiền xử lý dữ liệu        | Trích xuất đặc trưng           | Xử lý dữ liệu, huấn luyện tham số           | 100%           |


## Hoạt động nhóm: **[Các buổi họp](https://drive.google.com/drive/u/0/folders/13KqSeGFiZFktANKcmTRcsFp95Zxn7GKI)**

## Liên hệ
Nếu có thắc mắc, vui lòng liên hệ:  
- **Trương Thiên Ân** – an.truong241105@hcmut.edu.vn  
- **Lại Nguyễn Hoàng Hưng** – hung.lai2805@hcmut.edu.vn  
- **Nguyễn Tô Quốc Việt** – viet.nguyenluminous@hcmut.edu.vn  
