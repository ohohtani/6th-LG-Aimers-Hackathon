###앙상블(XG+CAT) 모델 하이퍼파라미터 튜닝(optuna)###
# 학습 및 컬레브레이션, 예측 실행은 앙상블_학습.py

import numpy as np
import pandas as pd
import optuna
import pickle
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

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

# 🔹 4. 범주형 변수 확인
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# 🔹 5. CATboost를 위한 objective 변수 문자열 변환
df_test_cat = df_test.copy()
df_test_cat = df_test_cat.drop(columns=["ID"], errors="ignore")
for col in cat_features:
    df_test_cat[col] = df_test_cat[col].astype(str)
X_cat = X.copy()
for col in cat_features:
    X_cat[col] = X_cat[col].astype(str)

# ✅ **XGBoost를 위한 레이블 인코딩 (Train & Test 합쳐서 진행)**
df_train_xgb = df_train.copy()
df_test_xgb = df_test.copy()

combined_df = pd.concat([df_train[cat_features], df_test[cat_features]], axis=0, ignore_index=True)

for col in cat_features:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col])

df_train_xgb[cat_features] = combined_df.iloc[:len(df_train)][cat_features]
df_test_xgb[cat_features] = combined_df.iloc[len(df_train):][cat_features]

# ✅ Train/Test 데이터 분할
X_xgb = df_train_xgb.drop(columns=["ID", target_col], errors="ignore")
X_test_xgb = df_test_xgb.drop(columns=["ID"], errors="ignore")

class_weights = {0: 0.2583, 1: 0.7417}  # 실패(0) -> 0.25, 성공(1) -> 0.75

positive_ct = ((y == 1).sum())
negative_ct = ((y == 0).sum())
scale_pos_weight = negative_ct/positive_ct

# 🔹 6. Optuna를 활용한 CatBoost & XGBoost 하이퍼파라미터 최적화
def objective(trial):
    print(f"🔄 현재 {trial.number + 1}번째 튜닝 진행 중...")
    
    # ✅ XGBoost 하이퍼파라미터
    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 500, 2000),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("xgb_learning_rate", 0.005, 0.1),
        "subsample": trial.suggest_uniform("xgb_subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("xgb_colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_loguniform("xgb_gamma", 1e-3, 10.0),
        "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_loguniform("xgb_reg_lambda", 1e-2, 100.0),
        "reg_alpha": trial.suggest_loguniform("xgb_reg_alpha", 1e-2, 100.0),
        "eval_metric": "auc",
        "missing": np.nan,
        "random_state": 42,
        "use_label_encoder": False,
        "early_stopping_rounds": 50,
        "scale_pos_weight": scale_pos_weight
    }

    # ✅ bootstrap_type 값을 먼저 고정
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bernoulli", "Poisson", "Bayesian"])
    
    # ✅ CatBoost 하이퍼파라미터
    cat_params = {
        "iterations": trial.suggest_int("cat_iterations", 500, 3000),
        "depth": trial.suggest_int("cat_depth", 4, 10),
        "learning_rate": trial.suggest_loguniform("cat_learning_rate", 0.005, 0.1),
        "l2_leaf_reg": trial.suggest_loguniform("cat_l2_leaf_reg", 1.0, 50.0),
        "border_count": trial.suggest_int("cat_border_count", 16, 64),
        "grow_policy": trial.suggest_categorical("cat_grow_policy", ["SymmetricTree", "Lossguide", "Depthwise"]),
        "bootstrap_type": bootstrap_type,
        "class_weights": [class_weights[0], class_weights[1]],
        "random_seed": 42,
        "task_type": "GPU",
        "eval_metric": "Logloss",
        "loss_function": "Logloss",
        "verbose": 0
    }

    # ✅ Stratified K-Fold 검증 적용
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, valid_idx in kf.split(X, y):
        X_train_xgb, X_valid_xgb = X_xgb.iloc[train_idx], X_xgb.iloc[valid_idx]
        X_train_cat, X_valid_cat = X_cat.iloc[train_idx], X_cat.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model_xgb = XGBClassifier(**xgb_params)
        model_cat = CatBoostClassifier(**cat_params)

        model_xgb.fit(X_train_xgb, y_train, eval_set=[(X_valid_xgb, y_valid)], verbose=0)
        model_cat.fit(X_train_cat, y_train, eval_set=(X_valid_cat, y_valid), cat_features=cat_features, verbose=0)

        y_pred_xgb = model_xgb.predict_proba(X_valid_xgb)[:, 1]
        y_pred_cat = model_cat.predict_proba(X_valid_cat)[:, 1]

        y_pred_ensemble = (y_pred_xgb + y_pred_cat) / 2
        auc_scores.append(roc_auc_score(y_valid, y_pred_ensemble))

    auc_score = np.mean(auc_scores)

    # ✅ 현재 trial의 AUC 점수 출력
    print(f"✅ {trial.number + 1}번째 튜닝 완료! ROC-AUC: {auc_score:.6f}")

    # ✅ 최고 점수 갱신 여부 확인 및 출력
    if trial.number == 0 or auc_score > study.best_value:
        print(f"🔥 New Best Trial Found! AUC: {auc_score:.6f} (Previous Best: {study.best_value if trial.number > 0 else 'None'})")

    return auc_score

# 🔹 7. Optuna 실행 (최적화)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# 🔹 8. 최적 파라미터 저장
best_params = study.best_params

# ✅ XGBoost 관련 파라미터만 추출
best_xgb_params = {k.replace("xgb_", ""): v for k, v in best_params.items() if "xgb_" in k}

# ✅ CatBoost 관련 파라미터만 추출
best_cat_params = {k.replace("cat_", ""): v for k, v in best_params.items() if "cat_" in k}

# ✅ CatBoost 추가 설정 (필수 설정값 추가)
best_cat_params["task_type"] = "GPU"
best_cat_params["eval_metric"] = "Logloss"
best_cat_params["loss_function"] = "Logloss"
best_cat_params["verbose"] = 100

# ✅ 최적화된 파라미터 저장
with open("best_xgb_params.pkl", "wb") as f:
    pickle.dump(best_xgb_params, f)
with open("best_cat_params.pkl", "wb") as f:
    pickle.dump(best_cat_params, f)

print(f"📁 최적화된 XGBoost & CatBoost 하이퍼파라미터가 구분되어 저장되었습니다.")
