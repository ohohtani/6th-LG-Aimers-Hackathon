# 📦 라이브러리 설치 및 불러오기
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from ngboost import NGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

# ✅ 데이터 검증 함수
def validate_data(X_train, X_test, cat_features):
    print("\n🚀 [데이터 검증 시작]")
    train_cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    test_cat_cols = X_test.select_dtypes(include=["object"]).columns.tolist()

    if train_cat_cols:
        print(f"🔎 학습 데이터 문자형 컬럼: {train_cat_cols}")
    else:
        print("✅ 학습 데이터에 문자형 컬럼 없음.")
    
    if test_cat_cols:
        print(f"🔎 테스트 데이터 문자형 컬럼: {test_cat_cols}")
    else:
        print("✅ 테스트 데이터에 문자형 컬럼 없음.")

    if X_train.isnull().values.any():
        print("❌ 학습 데이터 결측치 발견!")
    else:
        print("✅ 학습 데이터에 결측치 없음.")
        
    if X_test.isnull().values.any():
        print("❌ 테스트 데이터 결측치 발견!")
    else:
        print("✅ 테스트 데이터에 결측치 없음.")
    
    if list(X_train.columns) != list(X_test.columns):
        print("❌ 훈련/테스트 컬럼 불일치!")
        raise ValueError("🚫 컬럼 불일치 문제 해결 필요.")
    else:
        print("✅ 훈련/테스트 컬럼 일치.")
    print("✅ [데이터 검증 완료]\n")

# 🔹 1. 데이터 로드
df_train = pd.read_csv("train3_updated.csv")
df_test = pd.read_csv("test3_updated.csv")
df_sample_submission = pd.read_csv("sample_submission.csv")
test_ids = df_sample_submission["ID"]

# 🔹 2. Train 데이터 준비
target_col = "임신 성공 여부"
X = df_train.drop(columns=["ID", target_col], errors="ignore")
y = df_train[target_col]
X_test = df_test.drop(columns=["ID"], errors="ignore")

# 🔹 3. 범주형 변수 확인
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# ✅ 데이터 검증
validate_data(X, X_test, cat_features)

# 🔹 4. 숫자형 데이터만 추출 (LightGBM & NGBoost용)
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()
X_numeric = X[numeric_features]
X_test_numeric = X_test[numeric_features]

# 🔹 5. 클래스 가중치 설정
class_weights = {0: 0.2583, 1: 0.7417}

# ✅ 공통 검증 K-Fold 설정
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -------------------------------
# 🔹 6. CatBoost 튜닝
# -------------------------------
def catboost_objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.1),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1.0, 50.0),
        "border_count": trial.suggest_int("border_count", 16, 64),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide", "Depthwise"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bernoulli", "Poisson", "Bayesian"]),
        "class_weights": [class_weights[0], class_weights[1]],
        "random_seed": 42,
        "task_type": "GPU",
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "verbose": 0
    }

    if params["bootstrap_type"] in ["Bernoulli", "Poisson"]:
        params["subsample"] = trial.suggest_uniform("subsample", 0.7, 1.0)

    aucs = []
    for train_idx, valid_idx in kf.split(X, y):
        model = CatBoostClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], eval_set=(X.iloc[valid_idx], y.iloc[valid_idx]),
                  cat_features=cat_features, early_stopping_rounds=100, verbose=0)
        preds = model.predict_proba(X.iloc[valid_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[valid_idx], preds))
    return np.mean(aucs)

study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(catboost_objective, n_trials=20)
best_params_cat = study_cat.best_params
print("✅ CatBoost 최적 파라미터:", best_params_cat)

# -------------------------------
# 🔹 7. LightGBM 튜닝 (숫자형만 사용)
# -------------------------------
def lgbm_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-2, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-2, 10.0),
        "random_state": 42
    }

    aucs = []
    for train_idx, valid_idx in kf.split(X_numeric, y):
        model = LGBMClassifier(**params)
        model.fit(X_numeric.iloc[train_idx], y.iloc[train_idx],
                  eval_set=[(X_numeric.iloc[valid_idx], y.iloc[valid_idx])],
                  early_stopping_rounds=50, verbose=0)
        preds = model.predict_proba(X_numeric.iloc[valid_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[valid_idx], preds))
    return np.mean(aucs)

study_lgbm = optuna.create_study(direction="maximize")
study_lgbm.optimize(lgbm_objective, n_trials=20)
best_params_lgbm = study_lgbm.best_params
print("✅ LightGBM 최적 파라미터:", best_params_lgbm)

# -------------------------------
# 🔹 8. NGBoost 튜닝 (숫자형만 사용)
# -------------------------------
def ngboost_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
        "minibatch_frac": trial.suggest_uniform("minibatch_frac", 0.5, 1.0),
        "col_sample": trial.suggest_uniform("col_sample", 0.5, 1.0),
        "natural_gradient": trial.suggest_categorical("natural_gradient", [True, False]),
        "random_state": 42
    }

    aucs = []
    for train_idx, valid_idx in kf.split(X_numeric, y):
        model = NGBClassifier(**params)
        model.fit(X_numeric.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict_proba(X_numeric.iloc[valid_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[valid_idx], preds))
    return np.mean(aucs)

study_ngb = optuna.create_study(direction="maximize")
study_ngb.optimize(ngboost_objective, n_trials=20)
best_params_ngb = study_ngb.best_params
print("✅ NGBoost 최적 파라미터:", best_params_ngb)

# -------------------------------
# 🔹 9. 모델 학습
# -------------------------------
# ✅ CatBoost (범주형 + 숫자형)
cat_model = CatBoostClassifier(**best_params_cat)
cat_model.fit(X, y, cat_features=cat_features, verbose=100)

# ✅ LightGBM (숫자형만 사용)
lgbm_model = LGBMClassifier(**best_params_lgbm)
lgbm_model.fit(X_numeric, y)

# ✅ NGBoost (숫자형만 사용)
ngb_model = NGBClassifier(**best_params_ngb)
ngb_model.fit(X_numeric, y)

# -------------------------------
# 🔹 10. 테스트 예측 및 **가중치 적용 앙상블**
# -------------------------------
cat_preds = cat_model.predict_proba(X_test)[:, 1]
lgbm_preds = lgbm_model.predict_proba(X_test_numeric)[:, 1]
ngb_preds = ngb_model.predict_proba(X_test_numeric)[:, 1]

# ✅ CatBoost에 가중치 0.5, LightGBM과 NGBoost에 각각 0.25 적용
final_preds = (0.5 * cat_preds) + (0.25 * lgbm_preds) + (0.25 * ngb_preds)

# -------------------------------
# 🔹 11. 캘리브레이션 적용 (CatBoost 기준)
# -------------------------------
calibrated_preds = np.zeros(len(X_test))
for train_idx, valid_idx in kf.split(X, y):
    model = CatBoostClassifier(**best_params_cat)
    model.fit(X.iloc[train_idx], y.iloc[train_idx], cat_features=cat_features, verbose=0)

    calibrator = CalibratedClassifierCV(base_estimator=model, method='sigmoid', cv='prefit')
    calibrator.fit(X.iloc[valid_idx], y.iloc[valid_idx])
    calibrated_preds += calibrator.predict_proba(X_test)[:, 1] / kf.n_splits

# -------------------------------
# 🔹 12. 제출 파일 생성
# -------------------------------
df_submission = pd.DataFrame({"ID": test_ids, "probability": calibrated_preds})
df_submission.to_csv("ensemble_final.csv", index=False)
print("✅ 최종 제출 파일 저장 완료: 'ensemble_final.csv'")
