import numpy as np
import pandas as pd
import optuna
import torch
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# 🔹 1. 데이터 로드
file_path_train = "train_processed.csv"
file_path_test = "test_processed.csv"
sample_submission_path = "sample_submission.csv"

df_train = pd.read_csv(file_path_train)
df_test = pd.read_csv(file_path_test)
df_sample_submission = pd.read_csv(sample_submission_path)

# 🔹 2. 'ID' 컬럼 유지 (sample_submission을 위해 필요)
test_ids = df_sample_submission["ID"]

# 🔹 3. Train 데이터 준비
X = df_train.drop(columns=["ID", "임신 성공 여부"], errors="ignore")  
y = df_train["임신 성공 여부"]

# 🔹 4. Train 데이터 분할 (훈련 80%, 검증 20%)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 5. XGBoost 하이퍼파라미터 최적화 (GPU 사용)
def optimize_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.05),
        "subsample": trial.suggest_uniform("subsample", 0.3, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.3, 1.0),
        "gamma": trial.suggest_loguniform("gamma", 1e-6, 1e-1),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1.0, 20.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-5, 1e-1),
        "eval_metric": "auc",
        "tree_method": "gpu_hist",  # 🔥 GPU 사용
        "use_label_encoder": False,
        "random_state": 42
    }
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    preds = model.predict_proba(X_valid)[:, 1]
    return roc_auc_score(y_valid, preds)

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(optimize_xgb, n_trials=50)
best_xgb_params = study_xgb.best_params

# 🔹 6. TabNet 하이퍼파라미터 최적화 (GPU 사용)
def optimize_tabnet(trial):
    params = {
        "optimizer_params": dict(lr=trial.suggest_loguniform("learning_rate", 0.005, 0.05)),
        "batch_size": trial.suggest_int("batch_size", 128, 512),
        "virtual_batch_size": trial.suggest_int("virtual_batch_size", 64, 256),
    }
    model = TabNetClassifier(optimizer_fn=torch.optim.Adam, optimizer_params=params["optimizer_params"], device_name="cuda")  # 🔥 GPU 사용
    model.fit(X_train.values, y_train.values, eval_set=[(X_valid.values, y_valid.values)], eval_metric=["auc"], max_epochs=100, patience=10, batch_size=params["batch_size"], virtual_batch_size=params["virtual_batch_size"])
    preds = model.predict_proba(X_valid.values)[:, 1]
    return roc_auc_score(y_valid, preds)

study_tabnet = optuna.create_study(direction="maximize")
study_tabnet.optimize(optimize_tabnet, n_trials=50)
best_tabnet_params = study_tabnet.best_params

# 🔹 7. Classic Model (SVM, CatBoost) 최적화 및 학습
model_svm = SVC(probability=True, random_state=42)  # 🚫 SVM은 CPU에서만 실행 가능
model_catboost = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.01, random_state=42, verbose=False, task_type="GPU")  # 🔥 GPU 사용

# 🔹 8. 모델 학습
model_xgb = XGBClassifier(**best_xgb_params)
model_xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

model_tabnet = TabNetClassifier(optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=best_tabnet_params["learning_rate"]), device_name="cuda")  # 🔥 GPU 사용
model_tabnet.fit(X_train.values, y_train.values, eval_set=[(X_valid.values, y_valid.values)], eval_metric=["auc"], max_epochs=100, patience=10, batch_size=best_tabnet_params["batch_size"], virtual_batch_size=best_tabnet_params["virtual_batch_size"])

model_svm.fit(X_train, y_train)  # 🚫 CPU 전용
model_catboost.fit(X_train, y_train)

# 🔹 9. 예측값 생성
valid_preds_xgb = model_xgb.predict_proba(X_valid)[:, 1]
valid_preds_tabnet = model_tabnet.predict_proba(X_valid.values)[:, 1]
valid_preds_svm = model_svm.predict_proba(X_valid)[:, 1]
valid_preds_catboost = model_catboost.predict_proba(X_valid)[:, 1]

# 🔹 10. 가중 평균 앙상블 (Validation Loss 기반)
weights = {
    "xgb": 1 / (roc_auc_score(y_valid, valid_preds_xgb) + 1e-8),
    "tabnet": 1 / (roc_auc_score(y_valid, valid_preds_tabnet) + 1e-8),
    "svm": 1 / (roc_auc_score(y_valid, valid_preds_svm) + 1e-8),
    "catboost": 1 / (roc_auc_score(y_valid, valid_preds_catboost) + 1e-8),
}
total_weight = sum(weights.values())

valid_preds_ensemble = (
    (valid_preds_xgb * weights["xgb"]) +
    (valid_preds_tabnet * weights["tabnet"]) +
    (valid_preds_svm * weights["svm"]) +
    (valid_preds_catboost * weights["catboost"])
) / total_weight

roc_auc = roc_auc_score(y_valid, valid_preds_ensemble)
print(f"📊 최종 앙상블 ROC-AUC Score: {roc_auc:.4f}")

# 🔹 11. 테스트 데이터 예측
X_test = df_test.drop(columns=["ID"], errors="ignore")
test_preds_xgb = model_xgb.predict_proba(X_test)[:, 1]
test_preds_tabnet = model_tabnet.predict_proba(X_test.values)[:, 1]
test_preds_svm = model_svm.predict_proba(X_test)[:, 1]
test_preds_catboost = model_catboost.predict_proba(X_test)[:, 1]

test_preds_ensemble = (
    (test_preds_xgb * weights["xgb"]) +
    (test_preds_tabnet * weights["tabnet"]) +
    (test_preds_svm * weights["svm"]) +
    (test_preds_catboost * weights["catboost"])
) / total_weight

# 🔹 12. 최종 결과 저장
df_submission = pd.DataFrame({"ID": test_ids, "probability": test_preds_ensemble})
df_submission.to_csv("xgboost_tabnet_svm_catboost_ensemble.csv", index=False)

print(f"🚀 최적화된 앙상블 모델 결과 저장 완료!")
