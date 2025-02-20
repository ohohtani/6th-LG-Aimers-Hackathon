# 📦 라이브러리 설치 및 불러오기
import numpy as np
import pandas as pd
import optuna
import joblib
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV

# ✅ 옵튜나 인자 검증 함수
def validate_optuna_parameters():
    try:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: 1.0, n_trials=1)
        print("✅ Optuna 설정 확인 완료: 'n_trials' 인자 사용 가능합니다.")
    except TypeError as e:
        print("❌ 오류 발생:", e)

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
X = df_train.drop(columns=["ID", target_col], errors="ignore")
y = df_train[target_col]

# 🔹 3. 범주형 및 숫자형 변수 분리
cat_features = X.select_dtypes(include=["object"]).columns.tolist()
X_numeric = X.select_dtypes(exclude=["object"])
X_test_numeric = df_test.select_dtypes(exclude=["object"])

# 🔹 4. 데이터 분할 (Train / Calibration)
X_train, X_calib, y_train, y_calib = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_numeric_train = X_train.select_dtypes(exclude=["object"])
X_numeric_calib = X_calib.select_dtypes(exclude=["object"])

# 🔹 5. 클래스 가중치 및 scale_pos_weight 설정
class_weights = {0: 0.2583, 1: 0.7417}
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"✅ scale_pos_weight 계산 완료: {scale_pos_weight:.4f}")

# ✅ 옵튜나 인자 검증 실행
validate_optuna_parameters()

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

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=86)
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

# 🔹 7. XGBoost 하이퍼파라미터 최적화
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
        "scale_pos_weight": scale_pos_weight,
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

# 🔹 8. Optuna 최적화 실행
study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(catboost_objective, n_trials=30)
best_params_cat = study_cat.best_params
print("✅ CatBoost 최적 파라미터:", best_params_cat)

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(xgb_objective, n_trials=30)
best_params_xgb = study_xgb.best_params
print("✅ XGBoost 최적 파라미터:", best_params_xgb)

# 🔹 9. 최종 모델 학습 및 저장
cat_model = CatBoostClassifier(**best_params_cat, class_weights=[class_weights[0], class_weights[1]], task_type="GPU")
cat_model.fit(X_train, y_train, cat_features=cat_features, verbose=100)
joblib.dump(cat_model, "saved_cb_model.pkl")

xgb_model = xgb.XGBClassifier(**best_params_xgb)
xgb_model.fit(X_numeric_train, y_train, eval_set=[(X_numeric_calib, y_calib)], early_stopping_rounds=50, verbose=50)
joblib.dump(xgb_model, "saved_xgb_model.pkl")

print("✅ CatBoost & XGBoost 모델 학습 및 저장 완료!")

# 🔹 10. 소프트 보팅 확률 앙상블 (가중치 적용)
cb_model = joblib.load("saved_cb_model.pkl")
xgb_model = joblib.load("saved_xgb_model.pkl")

y_pred_cb_proba = cb_model.predict_proba(X_calib)[:, 1]
y_pred_xgb_proba = xgb_model.predict_proba(X_numeric_calib)[:, 1]

# ✔️ 가중치 적용 (CatBoost=7, XGBoost=3)
cat_weight, xgb_weight = 7, 3
total_weight = cat_weight + xgb_weight
y_pred_proba = (cat_weight * y_pred_cb_proba + xgb_weight * y_pred_xgb_proba) / total_weight
y_pred = (y_pred_proba >= 0.5).astype(int)

# 🔹 11. 성능 평가
auc_score = roc_auc_score(y_calib, y_pred_proba)
accuracy = accuracy_score(y_calib, y_pred)
print(f"✅ 가중치 적용 Soft Voting AUC: {auc_score:.4f}")
print(f"✅ 가중치 적용 Soft Voting Accuracy: {accuracy:.4f}")

# 🔹 12. 테스트 데이터 예측 및 제출 파일 생성
_y_pred_cb_proba = cb_model.predict_proba(df_test)[:, 1]
_y_pred_xgb_proba = xgb_model.predict_proba(X_test_numeric)[:, 1]

ensemble_pred_proba = (cat_weight * _y_pred_cb_proba + xgb_weight * _y_pred_xgb_proba) / total_weight

df_submission = pd.DataFrame({"ID": test_ids, "probability": ensemble_pred_proba})
submission_file_path = "weighted_soft_voting_submission.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ 가중치 적용 확률 기반 제출 파일이 '{submission_file_path}'로 저장되었습니다.")
