import numpy as np
import pandas as pd
import pickle
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# 🔹 1. 데이터 로드
file_path_train = "train3_updated.csv"
file_path_test = "test3_updated.csv"
sample_submission_path = "sample_submission.csv"

df_train = pd.read_csv(file_path_train)
df_test = pd.read_csv(file_path_test)
df_sample_submission = pd.read_csv(sample_submission_path)

# 🔹 2. 'ID' 컬럼 유지
test_ids = df_sample_submission["ID"]

# 🔹 3. 데이터 준비 (파생 변수 생성 포함)
target_col = "임신 성공 여부"
if target_col not in df_train.columns:
    raise ValueError(f"❌ '{target_col}' 컬럼이 존재하지 않습니다. 데이터 확인 필요!")

X = df_train.drop(columns=["ID", target_col], errors="ignore")
y = df_train[target_col]
X_test = df_test.drop(columns=["ID"], errors="ignore")

# ✅ 파생 변수 생성 함수 (1번 & 3번 적용)
def add_derived_features(df):
    df = df.copy()
    
    # 🔑 1번: 시술 당시 나이 수치형 변환
    age_map = {
        "만18-34세": 26, "만35-37세": 36, "만38-39세": 38, 
        "만40-42세": 41, "만43-44세": 43, "만45-50세": 47
    }
    df['시술 당시 나이_수치'] = df['시술 당시 나이'].map(age_map)
    
    # 🔑 3번: 나이 제곱 및 로그 변환
    df['시술 당시 나이_제곱'] = df['시술 당시 나이_수치'] ** 2
    df['시술 당시 나이_로그'] = np.log1p(df['시술 당시 나이_수치'])

    # 기존 파생 변수 유지
    df['이식된_배아_비율'] = df['이식된 배아 수'] / df['총 생성 배아 수'].replace(0, np.nan)
    df['미세주입_효율'] = df['미세주입 배아 이식 수'] / df['미세주입에서 생성된 배아 수'].replace(0, np.nan)
    df['배아_해동_이식_차이'] = df['배아 이식 경과일'] - df['배아 해동 경과일']
    df['불임_원인_합계'] = (df['불명확 불임 원인'] + df['불임 원인 - 난관 질환'] + 
                          df['불임 원인 - 남성 요인'] + df['불임 원인 - 배란 장애'] + 
                          df['불임 원인 - 자궁내막증'])
    trial_map = {'0회': 0, '1회': 1, '2회': 2, '3회': 3, '4회': 4, '5회': 5, '6회 이상': 6}
    df['총_시술_횟수_수치'] = df['총 시술 횟수'].map(trial_map)
    df['총_임신_횟수_수치'] = df['총 임신 횟수'].map(trial_map)
    df['시술당_임신_효율'] = df['총_임신_횟수_수치'] / df['총_시술_횟수_수치'].replace(0, np.nan)
    
    # 범주형 NaN 처리
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    
    return df

# 파생 변수 적용
X = add_derived_features(X)
X_test = add_derived_features(X_test)

# 🔹 4. 범주형 변수 확인
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# 🔹 5. 최적 파라미터 탐색 (Optuna)
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 3000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.005, 0.1),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1.0, 50.0),
        "border_count": trial.suggest_int("border_count", 16, 64),
        "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide", "Depthwise"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bernoulli", "Poisson", "Bayesian"]),
        "task_type": "GPU",
        "eval_metric": "AUC",
        "loss_function": "Logloss",
        "verbose": 0,
        "auto_class_weights": "Balanced"
    }
    if params["bootstrap_type"] in ["Bernoulli", "Poisson"]:
        params["subsample"] = trial.suggest_uniform("subsample", 0.7, 1.0)

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    auc_scores = []
    for train_idx, valid_idx in kf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=cat_features, 
                  early_stopping_rounds=100, verbose=0)
        valid_preds = model.predict_proba(X_valid)[:, 1]
        auc_scores.append(roc_auc_score(y_valid, valid_preds))
    return np.mean(auc_scores)

# Optuna 실행 및 새 .pkl 저장
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_params.update({
    "task_type": "GPU",
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "verbose": 0,
    "auto_class_weights": "Balanced"
})

new_params_save_path = "cat_self_ensemble.pkl"
with open(new_params_save_path, "wb") as f:
    pickle.dump(best_params, f)

print(f"📁 새 최적화된 하이퍼파라미터가 저장되었습니다: {new_params_save_path}")
print(f"🎯 새 하이퍼파라미터: {best_params}")

# 🔹 6. 저장된 파라미터 불러오기
with open(new_params_save_path, "rb") as f:
    best_params = pickle.load(f)

print(f"📁 저장된 하이퍼파라미터를 불러왔습니다: {new_params_save_path}")
print(f"🎯 불러온 하이퍼파라미터: {best_params}")

# 🔹 7. 검증 세트 분리 및 최적 random_seed 탐색
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

seed_range = range(0, 100)
val_scores = []

for seed in seed_range:
    best_params["random_seed"] = seed
    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features, early_stopping_rounds=100, verbose=0)
    val_preds = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, val_preds)
    val_scores.append((seed, score))
    print(f"Seed {seed} - ROC-AUC: {score:.5f}")

# 상위 3개 시드 선택
top_seeds = sorted(val_scores, key=lambda x: x[1], reverse=True)[:3]
top_seeds = [s[0] for s in top_seeds]
print(f"선택된 상위 3개 시드: {top_seeds}")

# 🔹 8. 상위 시드로 앙상블
test_preds_optimal = np.zeros(len(X_test))
best_params["verbose"] = 100

for seed in top_seeds:
    best_params["random_seed"] = seed
    model = CatBoostClassifier(**best_params)
    model.fit(X, y, cat_features=cat_features, verbose=100)
    test_preds_optimal += model.predict_proba(X_test)[:, 1] / len(top_seeds)

# 🔹 9. 제출 파일 생성
df_submission = pd.DataFrame({"ID": test_ids, "probability": test_preds_optimal})
submission_file_path = "cat_self_ensemble.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ CatBoost 최적 시드 앙상블 예측 결과가 '{submission_file_path}' 로 저장되었습니다.")
