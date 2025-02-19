import numpy as np
import pandas as pd
import optuna
import pickle
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# 🔹 1. 데이터 로드
file_path_train = "train3.csv"
file_path_test = "test3.csv"
sample_submission_path = "sample_submission.csv"

df_train = pd.read_csv(file_path_train)
df_test = pd.read_csv(file_path_test)
df_sample_submission = pd.read_csv(sample_submission_path)

# 🔹 2. 'ID' 컬럼 유지 (sample_submission을 위해 필요)
test_ids = df_sample_submission["ID"]

# 🔹 3. Train 데이터 준비
target_col = "임신 성공 여부"
if target_col not in df_train.columns:
    raise ValueError(f"❌ '{target_col}' 컬럼이 존재하지 않습니다. 데이터 확인 필요!")

X = df_train.drop(columns=["ID", target_col], errors="ignore")
y = df_train[target_col]

# 🔹 4. 범주형 변수 확인 (CatBoost에서 직접 사용 가능)
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# 🔹 5. 클래스 가중치 설정 (불균형 데이터 보정)
class_weights = {0: 0.25, 1: 0.75}  # 실패(0) -> 0.25, 성공(1) -> 0.75

# 🔹 6. Optuna를 활용한 하이퍼파라미터 최적화 (K-Fold 적용)
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 3000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.1),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1.0, 50.0),
        "border_count": trial.suggest_int("border_count", 16, 64),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide", "Depthwise"]),
        "class_weights": [class_weights[0], class_weights[1]],  # 가중치 적용
        "random_seed": 42,
        "task_type": "GPU",
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "verbose": 0
    }
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold 교차 검증
    auc_scores = []
    
    for train_idx, valid_idx in kf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=cat_features,
            early_stopping_rounds=100,
            verbose=0
        )
        
        valid_preds = model.predict_proba(X_valid)[:, 1]
        auc_scores.append(roc_auc_score(y_valid, valid_preds))
    
    return np.mean(auc_scores)  # K-Fold 평균 AUC 반환

# 🔹 7. Optuna 실행 (최적화)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)  

# 🔹 8. 최적 파라미터 저장 (`pkl` 파일)
best_params = study.best_params
best_params["random_seed"] = 42
best_params["task_type"] = "GPU"
best_params["eval_metric"] = "AUC"
best_params["loss_function"] = "Logloss"
best_params["verbose"] = 100

# GPU와 호환되지 않는 colsample_bylevel 제거
if "colsample_bylevel" in best_params:
    del best_params["colsample_bylevel"]

# 최적화된 하이퍼파라미터를 pkl 파일로 저장
params_save_path = "ㄱ_진짜_K_FOLD.pkl"
with open(params_save_path, "wb") as f:
    pickle.dump(best_params, f)

print(f"📁 최적화된 하이퍼파라미터가 저장되었습니다: {params_save_path}")
print(f"🎯 최적의 하이퍼파라미터: {best_params}")

# 🔹 9. K-Fold 기반 최종 학습 및 예측 (앙상블 적용)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # K=5
test_preds = np.zeros(len(df_test))  # 테스트 예측값 저장

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"🔄 Fold {fold + 1} 학습 중...")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # CatBoost 모델 학습
    model = CatBoostClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        cat_features=cat_features,
        early_stopping_rounds=100,
        verbose=100
    )

    # 검증 데이터 예측
    valid_preds = model.predict_proba(X_valid)[:, 1]
    fold_auc = roc_auc_score(y_valid, valid_preds)
    print(f"🎯 Fold {fold + 1} ROC-AUC: {fold_auc:.8f}")

    # 테스트 데이터 예측 (각 Fold 모델의 예측값을 누적하여 평균)
    test_preds += model.predict_proba(df_test.drop(columns=["ID"], errors="ignore"))[:, 1] / kf.n_splits

# 🔹 10. sample_submission 형식으로 변환
df_submission = pd.DataFrame({"ID": test_ids, "probability": test_preds})

# 🔹 11. 최종 CSV 파일 저장
submission_file_path = "ㄱ_진짜_K_FOLD.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ K-Fold 기반 최종 CatBoost 모델의 예측 결과가 '{submission_file_path}' 로 저장되었습니다.")
