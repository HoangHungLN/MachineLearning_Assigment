from typing import Dict, Any, Tuple
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy import sparse

def _is_sparse(x):
    return sparse.issparse(x)

def _scaler_for(x):
    # with_mean=False để không phá hỏng dữ liệu sparse
    return StandardScaler(with_mean=not _is_sparse(x))

def run_models(
    Xtr, ytr, Xva, yva, Xte, yte,
    model_params: Dict[str, Dict[str, Any]],
    print_reports: bool = True
):
    """
    Train & chọn model tốt nhất theo VAL accuracy, rồi đánh giá trên TEST.

    model_params: dict tên_mô_hình -> dict tham số
      Hỗ trợ các key:
        - 'NaiveBayes' -> MultinomialNB(**params)
        - 'LogisticRegression' -> LogisticRegression(**params)
        - 'LinearSVC' -> LinearSVC(**params)
        - 'SVC' -> SVC(**params)   # đặt probability=True nếu cần predict_proba

    Return:
      best_name, best_model, results
        results[model_name] = {'model': model, 'val_acc': float, 'test_acc': float}
    """
    results = {}

    # 1) Naive Bayes (chỉ hợp dữ liệu không âm)
    if 'NaiveBayes' in model_params:
        nb_params = model_params['NaiveBayes']
        nb = MultinomialNB(**nb_params)
        nb.fit(Xtr, ytr)
        yva_pred = nb.predict(Xva)
        val_acc = accuracy_score(yva, yva_pred)
        if print_reports:
            print(f"[NaiveBayes] VAL acc = {val_acc:.4f}")
        results['NaiveBayes'] = {'model': nb, 'val_acc': val_acc}

    # 2) Logistic Regression (nên scale)
    if 'LogisticRegression' in model_params:
        lr_params = model_params['LogisticRegression']
        lr = make_pipeline(_scaler_for(Xtr), LogisticRegression(**lr_params))
        lr.fit(Xtr, ytr)
        yva_pred = lr.predict(Xva)
        val_acc = accuracy_score(yva, yva_pred)
        if print_reports:
            print(f"[LogisticRegression] VAL acc = {val_acc:.4f}")
        results['LogisticRegression'] = {'model': lr, 'val_acc': val_acc}

    # 3) LinearSVC (nhanh, mạnh; không có predict_proba)
    if 'LinearSVC' in model_params:
        lsvm_params = model_params['LinearSVC']
        lsvm = make_pipeline(_scaler_for(Xtr), LinearSVC(**lsvm_params))
        lsvm.fit(Xtr, ytr)
        yva_pred = lsvm.predict(Xva)
        val_acc = accuracy_score(yva, yva_pred)
        if print_reports:
            print(f"[LinearSVC] VAL acc = {val_acc:.4f}")
        results['LinearSVC'] = {'model': lsvm, 'val_acc': val_acc}

    # 4) SVC (có xác suất nếu probability=True; chậm hơn LinearSVC)
    if 'SVC' in model_params:
        svc_params = model_params['SVC']
        svc = make_pipeline(_scaler_for(Xtr), SVC(**svc_params))
        svc.fit(Xtr, ytr)
        yva_pred = svc.predict(Xva)
        val_acc = accuracy_score(yva, yva_pred)
        yte_pred = svc.predict(Xte)
        test_acc = accuracy_score(yte, yte_pred)
        if print_reports:
            print(f"[SVC] VAL acc = {val_acc:.4f} | TEST acc = {test_acc:.4f}")
        results['SVC'] = {'model': svc, 'val_acc': val_acc, 'test_acc': test_acc}

    if not results:
        raise ValueError("Không có mô hình nào để chạy. Hãy truyền model_params hợp lệ.")

    # # Chọn best theo VAL accuracy
    # best_name = max(results, key=lambda k: results[k]['val_acc'])
    # best_model = results[best_name]['model']

    # # In báo cáo TEST chi tiết
    # if print_reports:
    #     print(f"\n=== Best model (VAL): {best_name} | VAL acc = {results[best_name]['val_acc']:.4f} ===")
    #     yte_pred = best_model.predict(Xte)
    #     print(f"[{best_name}] TEST accuracy = {accuracy_score(yte, yte_pred):.4f}")
    #     print(classification_report(yte, yte_pred, zero_division=0))

    return results

def evaluate_model_on_test(model, Xte, yte, model_name):
    """Evaluates a trained model on the test set and prints the report."""
    print(f"\n--- Đánh giá mô hình {model_name} tốt nhất trên tập Test ---")
    yte_pred = model.predict(Xte)
    test_accuracy = accuracy_score(yte, yte_pred)
    print(f"Test Accuracy: {test_accuracy}")
    print(classification_report(yte, yte_pred, zero_division=0))
    return test_accuracy