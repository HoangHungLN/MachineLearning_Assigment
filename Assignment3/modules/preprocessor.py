import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import vgg16, resnet50, efficientnet
from sklearn.model_selection import train_test_split

def preprocessing(
    image_dir,
    csv_path,
    model_name="ResNet50",
    image_size=(224, 224),
    batch_size=64,
    val_split=0.1,
    test_split=0.1,
    shuffle=True,
    seed=42,
):
    """
    Tiền xử lý ảnh từ file CSV:
      - Đọc ảnh từ thư mục, resize, chuẩn hóa bằng preprocess_input của pretrained model
      - Chia train / val / test
      - Xử lý theo batch
    Trả về:
        train_ds, val_ds, test_ds, class_names
    """

    # Đọc file nhãn 
    df = pd.read_csv(csv_path)
    df["label"] = df["label"].astype("category") #chuyển nhãn từ chuỗi sang categorical
    class_names = df["label"].cat.categories
    labels = df["label"].cat.codes
    image_paths = [os.path.join(image_dir, fname) for fname in df["filename"]] #đường dẫn tuyệt đối đến từng ảnh

    #Chia dữ liệu
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=test_split, stratify=labels, random_state=seed
    )
    val_ratio = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=seed
    )

    #Chọn preprocess phù hợp
    preprocess_map = {
        "VGG16": vgg16.preprocess_input,
        "ResNet50": resnet50.preprocess_input,
        "EfficientNetB0": efficientnet.preprocess_input,
    }
    if model_name not in preprocess_map:
        raise ValueError("model_name chỉ hỗ trợ: 'VGG16', 'ResNet50', 'EfficientNetB0'")
    preprocess_func = preprocess_map[model_name]

    #Hàm load và tiền xử lý
    def load_and_preprocess(img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        img = preprocess_func(img)
        return img, label

    #Hàm tạo dataset
    def make_dataset(paths, labels, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths), seed=seed)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_dataset(X_train, y_train, shuffle=shuffle)
    val_ds = make_dataset(X_val, y_val)
    test_ds = make_dataset(X_test, y_test)

    # 6 In thông tin
    print(f"Tổng số ảnh: {len(df)}")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Kích thước ảnh: {image_size}, Batch size: {batch_size}")
    print(f"Model preprocess: {model_name}")
    print(f"Số lớp: {len(class_names)}")

    return train_ds, val_ds, test_ds, class_names
