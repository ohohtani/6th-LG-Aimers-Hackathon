# 📦 라이브러리 설치 및 불러오기
import numpy as np
import pandas as pd
import optuna
import pickle
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from ngboost import NGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
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

# 🔹 4. 클래스 가중치 설정
class_weights = {0: 0.2583, 1: 0.7417}

# ✅ 공통 검증 K-Fold 설정
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -------------------------------
# 🔹 5. CatBoost 튜닝
# -------------------------------
def catboost_objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 1000, 3000),  # 범위 확장
        "depth": trial.suggest_int("depth", 4, 12),  # 깊이 범위 확장
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),  # 학습률 범위 세밀화
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 0.1, 100.0),  # L2 정규화 범위 확장
        "border_count": trial.suggest_int("border_count", 32, 128),  # 경계 수 범위 확장
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
        params["subsample"] = trial.suggest_uniform("subsample", 0.6, 1.0)  # 서브샘플링 범위 조정

    aucs = []
    for train_idx, valid_idx in kf.split(X, y):
        model = CatBoostClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], eval_set=(X.iloc[valid_idx], y.iloc[valid_idx]),
                  cat_features=cat_features, early_stopping_rounds=100, verbose=0)
        preds = model.predict_proba(X.iloc[valid_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[valid_idx], preds))
    return np.mean(aucs)

study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(catboost_objective, n_trials=50)  # 시도 횟수 증가
best_params_cat = study_cat.best_params
print("✅ CatBoost 최적 파라미터:", best_params_cat)

# -------------------------------
# 🔹 6. LightGBM 튜닝
# -------------------------------
def lgbm_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 3000),  # 범위 확장
        "max_depth": trial.suggest_int("max_depth", 3, 15),  # 깊이 범위 확장
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.0001, 0.1),  # 학습률 범위 세밀화
        "num_leaves": trial.suggest_int("num_leaves", 31, 511),  # 잎의 수 범위 확장
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 100.0),  # 알파 범위 확장
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 100.0),  # 람다 범위 확장
        "random_state": 42
    }

    aucs = []
    for train_idx, valid_idx in kf.split(X, y):
        model = LGBMClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx],
                  eval_set=[(X.iloc[valid_idx], y.iloc[valid_idx])],
                  early_stopping_rounds=50, verbose=0)
        preds = model.predict_proba(X.iloc[valid_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[valid_idx], preds))
    return np.mean(aucs)

study_lgbm = optuna.create_study(direction="maximize")
study_lgbm.optimize(lgbm_objective, n_trials=50)  # 시도 횟수 증가
best_params_lgbm = study_lgbm.best_params
print("✅ LightGBM 최적 파라미터:", best_params_lgbm)

# -------------------------------
# 🔹 7. NGBoost 튜닝
# -------------------------------
def ngboost_objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 3000),  # 범위 확장
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.0001, 0.1),  # 학습률 범위 세밀화
        "minibatch_frac": trial.suggest_uniform("minibatch_frac", 0.3, 1.0),  # 미니배치 비율 범위 조정
        "col_sample": trial.suggest_uniform("col_sample", 0.3, 1.0),  # 컬럼 샘플링 범위 조정
        "natural_gradient": trial.suggest_categorical("natural_gradient", [True, False]),
        "random_state": 42
    }

    aucs = []
    for train_idx, valid_idx in kf.split(X, y):
        model = NGBClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict_proba(X.iloc[valid_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[valid_idx], preds))
    return np.mean(aucs)

study_ngb = optuna.create_study(direction="maximize")
study_ngb.optimize(ngboost_objective, n_trials=50)  # 시도 횟수 증가
best_params_ngb = study_ngb.best_params
print("✅ NGBoost 최적 파라미터:", best_params_ngb)

# -------------------------------
# 🔹 8. 모델 학습
# -------------------------------
# CatBoost 모델 학습
cat_model = CatBoostClassifier(**best_params_cat)
cat_model.fit(X, y, cat_features=cat_features, verbose=100)

# LightGBM 모델 학습
lgbm_model = LGBMClassifier(**best_params_lgbm)
lgbm_model.fit(X, y)

# NGBoost 모델 학습
ngb_model = NGBClassifier(**best_params_ngb)
ngb_model.fit(X, y)

# 테스트 데이터 예측
cat_preds = cat_model.predict_proba(X_test)[:, 1]
lgbm_preds = lgbm_model.predict_proba(X_test)[:, 1]
ngb_preds = ngb_model.predict_proba(X_test)[:, 1]

# 소프트 보팅
final_preds = (cat_preds + lgbm_preds + ngb_preds) / 3

# -------------------------------
# 🔹 10. 캘리브레이션 적용
# -------------------------------
calibrated_preds = np.zeros(len(X_test))
for train_idx, valid_idx in kf.split(X, y):
    model = CatBoostClassifier(**best_params_cat)
    model.fit(X.iloc[train_idx], y.iloc[train_idx], cat_features=cat_features, verbose=0)

    calibrator = CalibratedClassifierCV(base_estimator=model, method='sigmoid', cv='prefit')
    calibrator.fit(X.iloc[valid_idx], y.iloc[valid_idx])
    calibrated_preds += calibrator.predict_proba(X_test)[:, 1] / kf.n_splits

# -------------------------------
# 🔹 11. 제출 파일 생성
# -------------------------------
df_submission = pd.DataFrame({"ID": test_ids, "probability": calibrated_preds})
df_submission.to_csv("ensemble_final.csv", index=False)
print("✅ 최종 제출 파일 저장 완료: 'ensemble_final.csv'")
