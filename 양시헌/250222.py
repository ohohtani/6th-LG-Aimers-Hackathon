import numpy as np
import pandas as pd
import optuna
import pickle
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

# 🔹 1. 데이터 로드
file_path_train = "train4_updated.csv"
file_path_test = "test4_updated.csv"
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
class_weights = {0: 0.2583, 1: 0.7417}  # 실패(0) -> 0.25, 성공(1) -> 0.75

# 🔹 6. Optuna를 활용한 하이퍼파라미터 최적화 (K-Fold 적용)
def objective(trial):
    # ✅ bootstrap_type 값을 먼저 고정
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bernoulli", "Poisson", "Bayesian", "MVS"])

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

    # ✅ bootstrap_type이 "Bernoulli"나 "Poisson"일 때만 subsample 추가
    if bootstrap_type in ["Bernoulli", "Poisson"]:
        params["subsample"] = trial.suggest_uniform("subsample", 0.7, 1.0)

    # Stratified K-Fold 검증
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
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

# 🔹 8. 최적 파라미터 저장 (`pkl` 파일)
best_params = study.best_params
best_params["random_seed"] = 42
best_params["task_type"] = "GPU"
best_params["eval_metric"] = "AUC"
best_params["loss_function"] = "Logloss"
best_params["verbose"] = 100

if "colsample_bylevel" in best_params:
    del best_params["colsample_bylevel"]

params_save_path = "cat_knn.pkl"
with open(params_save_path, "wb") as f:
    pickle.dump(best_params, f)

print(f"📁 최적화된 하이퍼파라미터가 저장되었습니다: {params_save_path}")
print(f"🎯 최적의 하이퍼파라미터: {best_params}")

# 🔹 9. 최적 하이퍼파라미터 적용하여 전체 데이터 학습
best_params["class_weights"] = [class_weights[0], class_weights[1]]

final_model = CatBoostClassifier(**best_params)
final_model.fit(
    X, y,  # 전체 데이터 사용
    cat_features=cat_features,
    verbose=100
)

# 🔹 10. 테스트 데이터 예측 (확률값 저장 - 원래 코드)
X_test = df_test.drop(columns=["ID"], errors="ignore")
test_preds = final_model.predict_proba(X_test)[:, 1]

# 🔹 11. ✅ 캘리브레이션 후처리 추가
print("🔄 캘리브레이션 적용 중...")

# 캘리브레이션을 위해 교차 검증을 사용
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
calibrated_preds = np.zeros(len(X_test))

for train_idx, valid_idx in kf.split(X, y):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # 기본 모델 복사 및 학습
    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, cat_features=cat_features, verbose=0)

    # 캘리브레이터 적용
    calibrator = CalibratedClassifierCV(estimator=model, method='sigmoid', cv='prefit')
    calibrator.fit(X_valid, y_valid)

    # 테스트 데이터에 대해 캘리브레이션 적용
    calibrated_preds += calibrator.predict_proba(X_test)[:, 1] / kf.n_splits

print("✅ 캘리브레이션 완료!")

# 🔹 12. sample_submission 형식으로 변환
df_submission = pd.DataFrame({"ID": test_ids, "probability": calibrated_preds})

# 🔹 13. 최종 CSV 파일 저장
submission_file_path = "cat_knn.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ 캘리브레이션된 CatBoost 모델의 예측 결과가 '{submission_file_path}' 로 저장되었습니다.")
