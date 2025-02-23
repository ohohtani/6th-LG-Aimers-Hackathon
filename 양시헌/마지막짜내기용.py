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
class_weights = {0: 0.2583, 1: 0.7417}  # 실패(0) -> 0.25, 성공(1) -> 0.75

# 🔹 6. Optuna를 활용한 하이퍼파라미터 최적화 (K-Fold 적용)
def objective(trial):
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Poisson", "Bayesian"])
    params = {
        "iterations": trial.suggest_int("iterations", 500, 5000),
        "depth": trial.suggest_int("depth", 4, 12),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.2),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 0.1, 100.0),
        "border_count": trial.suggest_int("border_count", 16, 128),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide", "Depthwise"]),
        "bootstrap_type": bootstrap_type,
        "bagging_temperature": trial.suggest_uniform("bagging_temperature", 0, 1),
        "random_strength": trial.suggest_loguniform("random_strength", 0.1, 10),
        "max_ctr_complexity": trial.suggest_int("max_ctr_complexity", 2, 8),
        "od_type": trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
        "od_wait": trial.suggest_int("od_wait", 50, 200),
        "class_weights": [class_weights[0], class_weights[1]],
        "random_seed": 42,
        "task_type": "GPU",
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "verbose": 0
    }
    if bootstrap_type == "Poisson":
        params["subsample"] = trial.suggest_uniform("subsample", 0.5, 1.0)

    # Stratified K-Fold 검증 (15개로 확장)
    kf = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, valid_idx in kf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=cat_features,
            early_stopping_rounds=300,  # Early Stopping 강화
            verbose=0
        )

        valid_preds = model.predict_proba(X_valid)[:, 1]
        auc_scores.append(roc_auc_score(y_valid, valid_preds))

    return np.mean(auc_scores)

# 🔹 7. Optuna 실행 (200회로 확장)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)

# 🔹 8. 최적 파라미터 저장 (`pkl` 파일)
best_params = study.best_params
best_params["random_seed"] = 42
best_params["task_type"] = "GPU"
best_params["eval_metric"] = "AUC"
best_params["loss_function"] = "Logloss"
best_params["verbose"] = 100

if "colsample_bylevel" in best_params:
    del best_params["colsample_bylevel"]

params_save_path = "cat_knn_optimized.pkl"
with open(params_save_path, "wb") as f:
    pickle.dump(best_params, f)

print(f"📁 최적화된 하이퍼파라미터가 저장되었습니다: {params_save_path}")
print(f"🎯 최적의 하이퍼파라미터: {best_params}")

# 🔹 9. 최적 하이퍼파라미터 적용하여 전체 데이터 학습 (최대 반복)
best_params["iterations"] = 10000  # 강제로 10000번 반복
best_params["class_weights"] = [class_weights[0], class_weights[1]]

final_model = CatBoostClassifier(**best_params)
final_model.fit(
    X, y,  # 전체 데이터 사용
    cat_features=cat_features,
    verbose=100  # Early Stopping 제거
)

# 🔹 10. 피처 중요도 기반 상위 피처 선택 및 재학습
feature_importance = final_model.get_feature_importance()
feature_names = X.columns
importance_df = pd.DataFrame({"feature": feature_names, "importance": feature_importance})
top_features = importance_df.sort_values("importance", ascending=False).head(20)["feature"].tolist()

# 상위 피처로 데이터 재구성
X_top = X[top_features]
X_test = df_test.drop(columns=["ID"], errors="ignore")
X_test_top = X_test[top_features]

# 상위 피처로 재학습
final_model.fit(
    X_top, y,
    cat_features=[f for f in cat_features if f in top_features],
    verbose=100
)

# 🔹 11. 테스트 데이터 예측 (기본 예측)
test_preds = final_model.predict_proba(X_test_top)[:, 1]

# 🔹 12. 캘리브레이션 후처리 (Sigmoid + Isotonic 평균)
print("🔄 캘리브레이션 적용 중...")
kf = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)  # 15개로 확장
calibrated_preds_sigmoid = np.zeros(len(X_test))
calibrated_preds_isotonic = np.zeros(len(X_test))

for train_idx, valid_idx in kf.split(X_top, y):
    X_train, X_valid = X_top.iloc[train_idx], X_top.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # 기본 모델 학습
    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, cat_features=[f for f in cat_features if f in top_features], verbose=0)

    # Sigmoid 캘리브레이션
    calibrator_sigmoid = CalibratedClassifierCV(estimator=model, method='sigmoid', cv='prefit')
    calibrator_sigmoid.fit(X_valid, y_valid)
    calibrated_preds_sigmoid += calibrator_sigmoid.predict_proba(X_test_top)[:, 1] / kf.n_splits

    # Isotonic 캘리브레이션
    calibrator_isotonic = CalibratedClassifierCV(estimator=model, method='isotonic', cv='prefit')
    calibrator_isotonic.fit(X_valid, y_valid)
    calibrated_preds_isotonic += calibrator_isotonic.predict_proba(X_test_top)[:, 1] / kf.n_splits

# 두 방식 평균
calibrated_preds = (calibrated_preds_sigmoid + calibrated_preds_isotonic) / 2
print("✅ 캘리브레이션 완료!")

# 🔹 13. sample_submission 형식으로 변환
df_submission = pd.DataFrame({"ID": test_ids, "probability": calibrated_preds})

# 🔹 14. 최종 CSV 파일 저장
submission_file_path = "cat_knn_optimized.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ 캘리브레이션된 CatBoost 모델의 예측 결과가 '{submission_file_path}' 로 저장되었습니다.")
