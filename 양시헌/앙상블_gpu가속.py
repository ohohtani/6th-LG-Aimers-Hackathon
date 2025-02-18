import os
import numpy as np
import pandas as pd
import optuna
import pickle
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# 🔹 1. GPU 1번 사용 강제 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # NVIDIA GeForce RTX (GPU 1번) 강제 사용

# 🔹 2. 데이터 로드
file_path_train = "train3_updated.csv"
file_path_test = "test3_updated.csv"
sample_submission_path = "sample_submission.csv"

df_train = pd.read_csv(file_path_train)
df_test = pd.read_csv(file_path_test)
df_sample_submission = pd.read_csv(sample_submission_path)

# 🔹 3. 'ID' 컬럼 유지
test_ids = df_sample_submission["ID"]

# 🔹 4. Train 데이터 준비
target_col = "임신 성공 여부"
if target_col not in df_train.columns:
    raise ValueError(f"❌ '{target_col}' 컬럼이 존재하지 않습니다.")

X = df_train.drop(columns=["ID", target_col], errors="ignore")
y = df_train[target_col]

# 🔹 5. 범주형 변수 확인
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# 🔹 6. XGBoost를 위한 Label Encoding
X_xgb = X.copy()
X_test_xgb = df_test.drop(columns=["ID"], errors="ignore")

if cat_features:
    combined_df = pd.concat([X[cat_features], X_test_xgb[cat_features]])
    for col in cat_features:
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col])
    
    X_xgb[cat_features] = combined_df.iloc[:len(X)][cat_features]
    X_test_xgb[cat_features] = combined_df.iloc[len(X):][cat_features]

# 🔹 7. 클래스 가중치 설정
class_weights = {0: 0.25, 1: 0.75}

# 🔹 8. Optuna를 활용한 하이퍼파라미터 최적화 (K-Fold 적용)
auc_history = []

def objective(trial):
    """Optuna를 이용한 XGBoost & CatBoost 하이퍼파라미터 최적화"""
    params_xgb = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 500, 3000),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("xgb_learning_rate", 0.005, 0.1),
        "subsample": trial.suggest_uniform("xgb_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("xgb_colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_loguniform("xgb_reg_lambda", 1.0, 50.0),
        "reg_alpha": trial.suggest_loguniform("xgb_reg_alpha", 0.01, 10.0),
        "eval_metric": "auc",
        "random_state": 10,
        "early_stopping_rounds": 100,
        "tree_method": "hist",
        "device": "cuda"  # GPU 사용
    }

    params_cat = {
        "iterations": trial.suggest_int("cat_iterations", 500, 3000),
        "depth": trial.suggest_int("cat_depth", 4, 10),
        "learning_rate": trial.suggest_loguniform("cat_learning_rate", 0.005, 0.1),
        "l2_leaf_reg": trial.suggest_loguniform("cat_l2_leaf_reg", 1.0, 50.0),
        "border_count": trial.suggest_int("cat_border_count", 16, 64),
        "grow_policy": trial.suggest_categorical("cat_grow_policy", ["SymmetricTree", "Lossguide", "Depthwise"]),
        "class_weights": [class_weights[0], class_weights[1]],
        "random_seed": 10,
        "task_type": "GPU",
        "devices": "1",  # GPU 1번 사용 설정
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "verbose": 0
    }

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)
    auc_scores = []

    for train_idx, valid_idx in kf.split(X, y):
        X_train_xgb, X_valid_xgb = X_xgb.iloc[train_idx], X_xgb.iloc[valid_idx]
        X_train_cat, X_valid_cat = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        # XGBoost 모델 학습
        model_xgb = XGBClassifier(**params_xgb)
        model_xgb.fit(X_train_xgb, y_train, eval_set=[(X_valid_xgb, y_valid)], verbose=0)

        # CatBoost 모델 학습
        model_cat = CatBoostClassifier(**params_cat)
        model_cat.fit(X_train_cat, y_train, eval_set=(X_valid_cat, y_valid), cat_features=cat_features, verbose=0)

        # Soft Voting
        preds_xgb = model_xgb.predict_proba(X_valid_xgb)[:, 1]
        preds_cat = model_cat.predict_proba(X_valid_cat)[:, 1]
        preds_ensemble = (preds_xgb + preds_cat) / 2

        auc_scores.append(roc_auc_score(y_valid, preds_ensemble))

    mean_auc = np.mean(auc_scores)
    auc_history.append(mean_auc)

    print(f"🟢 Trial {trial.number} | AUC Score: {mean_auc:.5f}")

    return mean_auc

# 🔹 9. Optuna 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# 🔹 10. 최적 파라미터 저장 (파일로 저장)
best_params = study.best_params
with open("ensemble_hyperparameters.pkl", "wb") as f:
    pickle.dump(best_params, f)

print("📁 최적 하이퍼파라미터가 'ensemble_hyperparameters.pkl' 파일에 저장되었습니다.")

# 🔹 11. 최적 모델 학습
best_params_xgb = {k.replace("xgb_", ""): v for k, v in best_params.items() if k.startswith("xgb_")}
best_params_cat = {k.replace("cat_", ""): v for k, v in best_params.items() if k.startswith("cat_")}

model_xgb = XGBClassifier(**best_params_xgb, tree_method="hist", device="cuda")
model_cat = CatBoostClassifier(**best_params_cat, cat_features=cat_features, task_type="GPU", devices='1')

model_xgb.fit(X_xgb, y)
model_cat.fit(X, y)

# 🔹 12. 테스트 데이터 예측
test_preds_xgb = model_xgb.predict_proba(X_test_xgb)[:, 1]
test_preds_cat = model_cat.predict_proba(df_test.drop(columns=["ID"], errors="ignore"))[:, 1]
test_preds_ensemble = (test_preds_xgb + test_preds_cat) / 2

# 🔹 13. 제출 파일 생성
df_submission = pd.DataFrame({"ID": test_ids, "probability": test_preds_ensemble})
submission_file_path = "ensemble_best2.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ 최적화된 결과가 '{submission_file_path}' 에 저장되었습니다.")
