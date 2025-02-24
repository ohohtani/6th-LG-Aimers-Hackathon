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

# 🔹 2. 'ID' 컬럼 유지
test_ids = df_sample_submission["ID"]

# 🔹 3. Train 데이터 준비
target_col = "임신 성공 여부"
if target_col not in df_train.columns:
    raise ValueError(f"❌ '{target_col}' 컬럼이 존재하지 않습니다. 데이터 확인 필요!")

X = df_train.drop(columns=["ID", target_col], errors="ignore")
y = df_train[target_col]
X_test = df_test.drop(columns=["ID"], errors="ignore")

# 🔹 4. 범주형 변수 확인
cat_features = X.select_dtypes(include=["object"]).columns.tolist()
print(f"📋 범주형 컬럼: {cat_features}")

# 🔹 5. 클래스 가중치 설정
class_weights = {0: 0.2583, 1: 0.7417}

# 🔹 6. Optuna 하이퍼파라미터 최적화 (K-Fold 검증 내부 적용)
def objective(trial):
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
        "task_type": "GPU",
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "verbose": 0
    }

    if bootstrap_type in ["Bernoulli", "Poisson"]:
        params["subsample"] = trial.suggest_uniform("subsample", 0.7, 1.0)

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

# 🔎 7. Optuna 최적화 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# 🔑 8. 최적 파라미터 저장 및 처리
best_params = study.best_params
best_params.update({
    "task_type": "GPU",
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "verbose": 100,
    "class_weights": [class_weights[0], class_weights[1]]
})

# ✅ random_seed 중복 제거
best_params.pop("random_seed", None)

params_save_path = "cat_after.pkl"
with open(params_save_path, "wb") as f:
    pickle.dump(best_params, f)

print(f"📁 최적화된 하이퍼파라미터가 저장되었습니다: {params_save_path}")
print(f"🎯 최적의 하이퍼파라미터: {best_params}")

# 🔹 9. 최적 파라미터로 전체 데이터 학습
final_model = CatBoostClassifier(**best_params, random_seed=42)  # 중복 문제 해결
final_model.fit(
    X, y,
    cat_features=cat_features,
    verbose=100
)

# 🔹 10. 테스트 데이터 예측
test_preds = final_model.predict_proba(X_test)[:, 1]

# 🔹 11. ✅ 캘리브레이션 적용
print("🔄 캘리브레이션 적용 중...")

calibrator = CalibratedClassifierCV(base_estimator=final_model, method='sigmoid', cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
calibrator.fit(X, y)
calibrated_preds = calibrator.predict_proba(X_test)[:, 1]

print("✅ 캘리브레이션 완료!")

# 🔹 12. 제출 파일 생성
df_submission = pd.DataFrame({"ID": test_ids, "probability": calibrated_preds})
submission_file_path = "cat_after_calibrated.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ 캘리브레이션된 예측 결과가 '{submission_file_path}' 로 저장되었습니다.")
