import numpy as np
import pandas as pd
import optuna
import pickle
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

# 🔹 1. 데이터 로드
file_path_train = "train3_updated.csv"
file_path_test = "test3_updated.csv"
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
class_weights = {0: 0.2583, 1: 0.7417}

# 🔹 6. Optuna를 활용한 하이퍼파라미터 최적화 (K-Fold 적용)
def objective(trial):
    # ✅ bootstrap_type 값을 먼저 고정
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bernoulli", "Poisson", "Bayesian"])

    params = {
        "iterations": trial.suggest_int("iterations", 500, 3000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.1),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1.0, 50.0),
        "border_count": trial.suggest_int("border_count", 16, 64),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide", "Depthwise"]),
        "bootstrap_type": bootstrap_type,
        "class_weights": [class_weights[0], class_weights[1]],
        "random_seed": 42,
        "task_type": "GPU",
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "verbose": 0
    }

    if bootstrap_type in ["Bernoulli", "Poisson"]:
        params["subsample"] = trial.suggest_uniform("subsample", 0.7, 1.0)

    # Stratified K-Fold 검증
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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

    return np.mean(auc_scores)


# 🔹 7. Optuna 실행 (최적화)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# 🔹 8. 최적 파라미터 저장
best_params = study.best_params
best_params["random_seed"] = 42
best_params["task_type"] = "GPU"
best_params["eval_metric"] = "AUC"
best_params["loss_function"] = "Logloss"
best_params["verbose"] = 100

if "colsample_bylevel" in best_params:
    del best_params["colsample_bylevel"]

print(f"🎯 최적의 하이퍼파라미터: {best_params}")

# 🔹 9. 최적 하이퍼파라미터 적용하여 K-Fold 기반 학습 수행 및 예측
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_preds = np.zeros(len(df_test))
calibrated_preds = np.zeros(len(df_test))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"🔄 K-Fold {fold + 1} 학습 중...")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # 개별 K-Fold 모델 학습
    model = CatBoostClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        cat_features=cat_features,
        early_stopping_rounds=100,
        verbose=100
    )

    # 개별 Fold의 테스트 예측값 저장 (앙상블 평균)
    test_preds += model.predict_proba(df_test.drop(columns=["ID"], errors="ignore"))[:, 1] / kf.n_splits

    # 캘리브레이션 적용
    print(f"🔄 K-Fold {fold + 1} 캘리브레이션 적용 중...")
    calibrator = CalibratedClassifierCV(base_estimator=model, method='sigmoid', cv='prefit')
    calibrator.fit(X_valid, y_valid)

    # 캘리브레이션된 테스트 예측값 저장 (앙상블 평균)
    calibrated_preds += calibrator.predict_proba(df_test.drop(columns=["ID"], errors="ignore"))[:, 1] / kf.n_splits

print("✅ K-Fold 기반 최종 모델 학습 완료!")

# 🔹 10. sample_submission 형식으로 변환
df_submission = pd.DataFrame({"ID": test_ids, "probability": calibrated_preds})

# 🔹 11. 최종 CSV 파일 저장
submission_file_path = "catboost_kfold_calibrated.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ K-Fold 캘리브레이션된 CatBoost 모델의 예측 결과가 '{submission_file_path}' 로 저장되었습니다.")
