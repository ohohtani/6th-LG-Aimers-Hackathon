# 📦 라이브러리 설치 및 불러오기
import numpy as np
import pandas as pd
import optuna
import pickle
import xgboost as xgb
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

# 🔹 3. 범주형 및 숫자형 변수 분리
cat_features = X.select_dtypes(include=["object"]).columns.tolist()
X_numeric = X.select_dtypes(exclude=["object"])
X_test_numeric = df_test.select_dtypes(exclude=["object"])

# 🔹 4. 데이터 분할 (Train / Calibration / Test)
X_train_full, X_test_split, y_train_full, y_test_split = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_calib, y_train, y_calib = train_test_split(
    X_train_full, y_train_full, test_size=0.25, stratify=y_train_full, random_state=42
)

X_numeric_train = X_train.select_dtypes(exclude=["object"])
X_numeric_calib = X_calib.select_dtypes(exclude=["object"])

# 🔹 5. 클래스 가중치 설정
class_weights = {0: 0.2583, 1: 0.7417}

# 🔹 6. CatBoost 하이퍼파라미터 최적화
def catboost_objective(trial):
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bernoulli", "Poisson", "Bayesian"])
    params = {
        "iterations": trial.suggest_int("iterations", 500, 1500),
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

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=86)
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

        preds = model.predict_proba(X_valid_fold)[:, 1]
        auc_scores.append(roc_auc_score(y_valid_fold, preds))

    return np.mean(auc_scores)

study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(catboost_objective, n_trials=50)
best_params_cat = study_cat.best_params
best_params_cat.update({
    "random_seed": 42,
    "task_type": "GPU",
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "verbose": 100,
    "class_weights": [class_weights[0], class_weights[1]]
})

# 🔹 7. XGBoost 하이퍼파라미터 최적화
def xgb_objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.1),
        "subsample": trial.suggest_uniform("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.7, 1.0),
        "lambda": trial.suggest_loguniform("lambda", 1e-3, 10.0),
        "alpha": trial.suggest_loguniform("alpha", 1e-3, 10.0),
        "seed": 42
    }

    dtrain = xgb.DMatrix(X_numeric_train, label=y_train)
    dvalid = xgb.DMatrix(X_numeric_calib, label=y_calib)

    booster = xgb.train(
        params, dtrain, num_boost_round=1000,
        evals=[(dvalid, "validation")],
        early_stopping_rounds=100, verbose_eval=False
    )

    preds = booster.predict(dvalid)
    return roc_auc_score(y_calib, preds)

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(xgb_objective, n_trial=50)
best_params_xgb = study_xgb.best_params

# 🔹 8. 최종 모델 학습 및 저장
catboost_model_path = "ensemble_cat_model.pkl"
xgboost_model_path = "ensemble_xgb_model.pkl"

# CatBoost 학습 및 저장
final_cat_model = CatBoostClassifier(**best_params_cat)
final_cat_model.fit(X_train, y_train, cat_features=cat_features, verbose=100)
with open(catboost_model_path, "wb") as f:
    pickle.dump(final_cat_model, f)
print(f"✅ CatBoost 모델이 '{catboost_model_path}'로 저장되었습니다.")

# XGBoost 학습 및 저장
dtrain_final = xgb.DMatrix(X_numeric_train, label=y_train)
final_xgb_model = xgb.train(best_params_xgb, dtrain_final, num_boost_round=500)
with open(xgboost_model_path, "wb") as f:
    pickle.dump(final_xgb_model, f)
print(f"✅ XGBoost 모델이 '{xgboost_model_path}'로 저장되었습니다.")

# 🔹 9. 모델 불러오기 및 캘리브레이션 적용
with open(catboost_model_path, "rb") as f:
    final_cat_model = pickle.load(f)

with open(xgboost_model_path, "rb") as f:
    final_xgb_model = pickle.load(f)

calibrator_cat = CalibratedClassifierCV(estimator=final_cat_model, method='isotonic', cv='prefit')
calibrator_cat.fit(X_calib, y_calib)

# 🔹 10. 예측
X_test_full = df_test.drop(columns=["ID"], errors="ignore")
cat_preds = calibrator_cat.predict_proba(X_test_full)[:, 1]
dtest = xgb.DMatrix(X_test_numeric)
xgb_preds = final_xgb_model.predict(dtest)

# 🔹 11. 앙상블 예측 (가중 평균)
cat_auc = study_cat.best_value
xgb_auc = study_xgb.best_value
cat_weight = cat_auc / (cat_auc + xgb_auc)
xgb_weight = xgb_auc / (cat_auc + xgb_auc)

ensemble_preds = (cat_weight * cat_preds) + (xgb_weight * xgb_preds)

# 🔹 12. 결과 저장
df_submission = pd.DataFrame({"ID": test_ids, "probability": ensemble_preds})
submission_file_path = "ensemble_submission.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ 최종 제출 파일이 '{submission_file_path}'로 저장되었습니다.")
