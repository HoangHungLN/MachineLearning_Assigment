from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def run_models(Xtr, ytr, Xva, yva, Xte, yte, model_params):
    """
    Huấn luyện và đánh giá các mô hình phân loại: Naive Bayes, Logistic Regression, SVM.

    Args:
        Xtr (sparse matrix or ndarray): Ma trận đặc trưng tập huấn luyện.
        ytr (ndarray): Nhãn tập huấn luyện.
        Xva (sparse matrix or ndarray): Ma trận đặc trưng tập validation.
        yva (ndarray): Nhãn tập validation.
        Xte (sparse matrix or ndarray): Ma trận đặc trưng tập kiểm tra.
        yte (ndarray): Nhãn tập kiểm tra.
        model_params (dict): Từ điển chứa tham số cho từng mô hình.
                             Ví dụ: {'NaiveBayes': {'alpha': 1.0},
                                      'LogisticRegression': {'C': 1.0, 'max_iter': 1000},
                                      'SVM': {'C': 1.0},
                             Chỉ các mô hình có trong dictionary này mới được huấn luyện.
    """
    results = {}

    if 'NaiveBayes' in model_params:
        nb_params = model_params['NaiveBayes']
        nb_model = MultinomialNB(**nb_params)
        nb_model.fit(Xtr, ytr)
        yva_pred_nb = nb_model.predict(Xva)
        val_accuracy = accuracy_score(yva, yva_pred_nb)
        print(f"Validation Accuracy: {val_accuracy}")
        results['NaiveBayes'] = {'model': nb_model, 'val_accuracy': val_accuracy}


    if 'LogisticRegression' in model_params:
        lr_params = model_params['LogisticRegression']
        lr_model = LogisticRegression(**lr_params)
        lr_model.fit(Xtr, ytr)
        yva_pred_lr = lr_model.predict(Xva)
        val_accuracy = accuracy_score(yva, yva_pred_lr)
        print(f"Validation Accuracy: {val_accuracy}")
        results['LogisticRegression'] = {'model': lr_model, 'val_accuracy': val_accuracy}


    if 'SVM' in model_params:
        svm_params = model_params['SVM']
        svm_model = LinearSVC(**svm_params)
        svm_model.fit(Xtr, ytr)
        yva_pred_svm = svm_model.predict(Xva)
        val_accuracy = accuracy_score(yva, yva_pred_svm)
        print(f"Validation Accuracy: {val_accuracy}")
        results['SVM'] = {'model': svm_model, 'val_accuracy': val_accuracy}

    return results

def evaluate_model_on_test(model, Xte, yte, model_name):
    """Evaluates a trained model on the test set and prints the report."""
    print(f"\n--- Đánh giá mô hình {model_name} tốt nhất trên tập Test ---")
    yte_pred = model.predict(Xte)
    test_accuracy = accuracy_score(yte, yte_pred)
    print(f"Test Accuracy: {test_accuracy}")
    print(classification_report(yte, yte_pred, zero_division=0))
    return test_accuracy
