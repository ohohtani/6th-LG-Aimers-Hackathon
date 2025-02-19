import numpy as np
import pandas as pd
import optuna
import pickle
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV

# 🔹 1. 데이터 로드
file_path_train = "train3_updated.csv"
file_path_test = "test3_updated.csv"
sample_submission_path = "sample_submission.csv"

df_train = pd.read_csv(file_path_train)
df_test = pd.read_csv(file_path_test)
df_sample_submission = pd.read_csv(sample_submission_path)

test_ids = df_sample_submission["ID"]

# 🔹 2. Train 데이터 준비
target_col = "임신 성공 여부"
if target_col not in df_train.columns:
    raise ValueError(f"❌ '{target_col}' 컬럼이 존재하지 않습니다. 데이터 확인 필요!")

X = df_train.drop(columns=["ID", target_col], errors="ignore")
y = df_train[target_col]

# 🔹 3. 범주형 변수 확인
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# 🔹 4. 데이터 분할 (Train / Calibration / Test)
X_train_full, X_test_split, y_train_full, y_test_split = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=42
)
# 👉 최종적으로: Train(60%) / Calibration(20%) / Test(20%)

# 🔹 5. 클래스 가중치 설정
class_weights = {0: 0.2583, 1: 0.7417}

# 🔹 6. Optuna를 활용한 하이퍼파라미터 최적화
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
        "random_seed": 42,
        "task_type": "GPU",
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "verbose": 0
    }

    if bootstrap_type in ["Bernoulli", "Poisson"]:
        params["subsample"] = trial.suggest_uniform("subsample", 0.7, 1.0)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, valid_idx in kf.split(X_train, y_train):
        X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=(X_valid_fold, y_valid_fold),
            cat_features=cat_features,
            early_stopping_rounds=100,
            verbose=0
        )

        valid_preds = model.predict_proba(X_valid_fold)[:, 1]
        auc_scores.append(roc_auc_score(y_valid_fold, valid_preds))

    return np.mean(auc_scores)

# 🔹 7. Optuna 실행 (최적화)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_params.update({
    "random_seed": 42,
    "task_type": "GPU",
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "verbose": 100,
    "class_weights": [class_weights[0], class_weights[1]]
})

# 🔹 8. 최적 파라미터 저장
params_save_path = "cat_after.pkl"
with open(params_save_path, "wb") as f:
    pickle.dump(best_params, f)

print(f"📁 최적화된 하이퍼파라미터가 저장되었습니다: {params_save_path}")
print(f"🎯 최적의 하이퍼파라미터: {best_params}")

# 🔹 9. 최종 모델 학습 (Train 데이터 사용)
final_model = CatBoostClassifier(**best_params)
final_model.fit(
    X_train, y_train,
    cat_features=cat_features,
    verbose=100
)

# 🔹 10. 캘리브레이션 적용 (Calibration 데이터 사용)
print("🔄 캘리브레이션 적용 중...")

calibrator = CalibratedClassifierCV(base_estimator=final_model, method='sigmoid', cv='prefit')
calibrator.fit(X_calib, y_calib)

print("✅ 캘리브레이션 완료!")

# 🔹 11. 테스트 데이터 예측 (캘리브레이션된 모델 사용)
X_test = df_test.drop(columns=["ID"], errors="ignore")
calibrated_preds = calibrator.predict_proba(X_test)[:, 1]

# 🔹 12. sample_submission 형식으로 변환
df_submission = pd.DataFrame({"ID": test_ids, "probability": calibrated_preds})

# 🔹 13. 최종 CSV 파일 저장
submission_file_path = "cat_after.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ 최종 제출 파일이 '{submission_file_path}'로 저장되었습니다.")
