# 📦 라이브러리 설치 및 불러오기
import numpy as np
import pandas as pd
import optuna
import pickle
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# 🔹 데이터 로드
df_train = pd.read_csv("train3_updated.csv")
df_test = pd.read_csv("test3_updated.csv")
df_sample_submission = pd.read_csv("sample_submission.csv")
test_ids = df_sample_submission["ID"]

target_col = "임신 성공 여부"
X = df_train.drop(columns=["ID", target_col], errors="ignore")
y = df_train[target_col]
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# ✅ lane_count로 데이터 분할
X1, X2, X3 = [X[X['lane_count'] == i].drop(['lane_count'], axis=1) for i in range(1, 4)]
y1, y2, y3 = [y[X['lane_count'] == i] for i in range(1, 4)]

target = df_test.drop(columns=["ID"])
target1, target2, target3 = [target[target['lane_count'] == i].drop(['lane_count'], axis=1) for i in range(1, 4)]

# ✅ StratifiedKFold 설정
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# ✅ 클래스 가중치 계산
class_weights = {0: 0.2583, 1: 0.7417}

# 🔹 6. CatBoost 하이퍼파라미터 최적화
def catboost_objective(trial, X_data, y_data):
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

    auc_scores = []
    for train_idx, valid_idx in skf.split(X_data, y_data):
        model = CatBoostClassifier(**params)
        model.fit(
            X_data.iloc[train_idx], y_data.iloc[train_idx],
            eval_set=(X_data.iloc[valid_idx], y_data.iloc[valid_idx]),
            cat_features=cat_features,
            early_stopping_rounds=100,
            verbose=0
        )
        preds = model.predict_proba(X_data.iloc[valid_idx])[:, 1]
        auc_scores.append(roc_auc_score(y_data.iloc[valid_idx], preds))

    return np.mean(auc_scores)

# 🔹 7. XGBoost 하이퍼파라미터 최적화 (scale_pos_weight 적용)
def xgb_objective(trial, X_data, y_data):
    positive_ct = (y_data == 1).sum()
    negative_ct = (y_data == 0).sum()
    scale_pos_weight = negative_ct / positive_ct

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
        "tree_method": "gpu_hist",
        "device": "cuda",
        "missing": np.nan,
        "random_state": 42,
        "use_label_encoder": False,
        "early_stopping_rounds": 30
    }

    auc_scores = []
    for train_idx, valid_idx in skf.split(X_data, y_data):
        dtrain = xgb.DMatrix(X_data.iloc[train_idx], label=y_data.iloc[train_idx])
        dvalid = xgb.DMatrix(X_data.iloc[valid_idx], label=y_data.iloc[valid_idx])

        booster = xgb.train(
            params, dtrain, num_boost_round=1000,
            evals=[(dvalid, "validation")],
            early_stopping_rounds=100, verbose_eval=False
        )

        preds = booster.predict(dvalid)
        auc_scores.append(roc_auc_score(y_data.iloc[valid_idx], preds))

    return np.mean(auc_scores)

# ✅ Optuna 실행 및 최적 파라미터 탐색
def tune_model(X_data, y_data):
    study_cat = optuna.create_study(direction="maximize")
    study_cat.optimize(lambda trial: catboost_objective(trial, X_data, y_data), n_trials=20)
    best_cat_params = study_cat.best_params

    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(lambda trial: xgb_objective(trial, X_data, y_data), n_trials=20)
    best_xgb_params = study_xgb.best_params

    return best_cat_params, best_xgb_params

# ✅ 모델 학습 및 예측 함수
def train_and_predict(X_data, y_data, target_data, best_cat_params, best_xgb_params):
    cat_preds, xgb_preds = np.zeros(len(target_data)), np.zeros(len(target_data))

    for train_idx, valid_idx in skf.split(X_data, y_data):
        # CatBoost
        cat_model = CatBoostClassifier(**best_cat_params, cat_features=cat_features, verbose=0)
        cat_model.fit(X_data.iloc[train_idx], y_data.iloc[train_idx])
        cat_preds += cat_model.predict_proba(target_data)[:, 1] / skf.n_splits

        # XGBoost
        dtrain = xgb.DMatrix(X_data.iloc[train_idx], label=y_data.iloc[train_idx])
        dtarget = xgb.DMatrix(target_data)
        xgb_model = xgb.train(best_xgb_params, dtrain, num_boost_round=500)
        xgb_preds += xgb_model.predict(dtarget) / skf.n_splits

    return cat_preds, xgb_preds

# ✅ 각 lane_count별 모델 튜닝 및 예측
best_cat1, best_xgb1 = tune_model(X1, y1)
cat_pred1, xgb_pred1 = train_and_predict(X1, y1, target1, best_cat1, best_xgb1)

best_cat2, best_xgb2 = tune_model(X2, y2)
cat_pred2, xgb_pred2 = train_and_predict(X2, y2, target2, best_cat2, best_xgb2)

best_cat3, best_xgb3 = tune_model(X3, y3)
cat_pred3, xgb_pred3 = train_and_predict(X3, y3, target3, best_cat3, best_xgb3)

# ✅ 앙상블 및 제출 파일 저장 (CatBoost: 70%, XGBoost: 30%)
submission = df_sample_submission.copy()
submission.loc[target1.index, 'probability'] = cat_pred1 * 0.70 + xgb_pred1 * 0.30
submission.loc[target2.index, 'probability'] = cat_pred2 * 0.70 + xgb_pred2 * 0.30
submission.loc[target3.index, 'probability'] = cat_pred3 * 0.70 + xgb_pred3 * 0.30

submission.to_csv("ensemble_submission_lane_count.csv", index=False)
print("✅ 최종 제출 파일이 'ensemble_submission_lane_count.csv'로 저장되었습니다.")
