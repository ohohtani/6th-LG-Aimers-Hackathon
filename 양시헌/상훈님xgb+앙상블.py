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

# ✅ 옵튜나 최적화 전 인자 검증 함수 (사전 오류 방지)
def validate_optuna_parameters():
    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: 1.0, n_trials=1)  # 올바른 인자 사용 확인
        print("✅ Optuna 설정 확인 완료: 'n_trials' 인자 사용 가능합니다.")
    except TypeError as e:
        print("❌ 오류 발생:", e)
        print("🔔 'n_trials' 인자를 확인해주세요!")

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

# ✅ XGBoost의 scale_pos_weight 계산 (클래스 불균형 처리)
positive_ct = (y_train == 1).sum()
negative_ct = (y_train == 0).sum()
scale_pos_weight = negative_ct / positive_ct
print(f"✅ scale_pos_weight 계산 완료: {scale_pos_weight:.4f}")

# 🔹 6. CatBoost 하이퍼파라미터 최적화
def catboost_objective(trial):
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bernoulli", "Poisson", "Bayesian"])
    params = {
        "iterations": trial.suggest_int("iterations", 500, 1500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
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
        params["subsample"] = trial.suggest_float("subsample", 0.7, 1.0)

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

# 🔹 7. XGBoost 하이퍼파라미터 최적화 (scale_pos_weight 적용)
def xgb_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 100.0, log=True),
        "scale_pos_weight": scale_pos_weight,  # ✅ 클래스 불균형 처리 추가
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda",
        "missing": np.nan,
        "random_state": 42,
        "use_label_encoder": False,
        "early_stopping_rounds": 30
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

# ✅ 옵튜나 인자 검증 실행
validate_optuna_parameters()

# 🔹 CatBoost 최적화 실행
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

# 🔹 XGBoost 최적화 실행
study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(xgb_objective, n_trials=50)
best_params_xgb = study_xgb.best_params

# 🔹 8. 최종 모델 학습 및 저장
catboost_model_path = "ensemble_cat_model.pkl"
xgboost_model_path = "ensemble_xgb_model.pkl"

# ✅ CatBoost 모델 학습 및 저장
final_cat_model = CatBoostClassifier(**best_params_cat)
final_cat_model.fit(X_train, y_train, cat_features=cat_features, verbose=100)
with open(catboost_model_path, "wb") as f:
    pickle.dump(final_cat_model, f)
print(f"✅ CatBoost 모델이 '{catboost_model_path}'로 저장되었습니다.")

# ✅ XGBoost 모델 학습 및 저장
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

# ✅ 교차 검증 기반 캘리브레이션 적용
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cat_calibrated_preds = np.zeros(len(X_test_numeric))
xgb_calibrated_preds = np.zeros(len(X_test_numeric))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_full, y_train_full), 1):
    print(f"📝 Fold {fold} 처리 중...")

    X_fold_train, X_fold_valid = X_train_full.iloc[train_idx], X_train_full.iloc[valid_idx]
    y_fold_train, y_fold_valid = y_train_full.iloc[train_idx], y_train_full.iloc[valid_idx]

    # ✅ CatBoost 교차 검증 기반 캘리브레이션
    cat_model_fold = CatBoostClassifier(**final_cat_model.get_params())
    cat_model_fold.fit(X_fold_train, y_fold_train, cat_features=cat_features, verbose=0)
    cat_calibrator = CalibratedClassifierCV(estimator=cat_model_fold, method='sigmoid', cv='prefit')
    cat_calibrator.fit(X_fold_valid, y_fold_valid)
    cat_calibrated_preds += cat_calibrator.predict_proba(X_test_numeric)[:, 1] / kf.n_splits

    # ✅ XGBoost 교차 검증 기반 캘리브레이션
    dtrain_fold = xgb.DMatrix(X_fold_train, label=y_fold_train)
    dvalid_fold = xgb.DMatrix(X_fold_valid, label=y_fold_valid)
    xgb_model_fold = xgb.train(best_params_xgb, dtrain_fold, num_boost_round=500)
    xgb_calibrator = CalibratedClassifierCV(estimator=xgb_model_fold, method='sigmoid', cv='prefit')
    xgb_calibrator.fit(X_fold_valid, y_fold_valid)
    xgb_calibrated_preds += xgb_calibrator.predict_proba(X_test_numeric)[:, 1] / kf.n_splits

print("✅ 교차 검증 캘리브레이션 완료!")

# 🔹 10. 앙상블 예측 (가중 평균)
cat_auc = study_cat.best_value
xgb_auc = study_xgb.best_value
cat_weight = cat_auc / (cat_auc + xgb_auc)
xgb_weight = xgb_auc / (cat_auc + xgb_auc)
ensemble_preds = (cat_weight * cat_calibrated_preds) + (xgb_weight * xgb_calibrated_preds)

# 🔹 11. 결과 저장
df_submission = pd.DataFrame({"ID": test_ids, "probability": ensemble_preds})
submission_file_path = "ensemble_submission_with_cv_calibration.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ 최종 제출 파일이 '{submission_file_path}'로 저장되었습니다.")
