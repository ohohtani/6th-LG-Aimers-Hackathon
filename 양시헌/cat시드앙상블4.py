import numpy as np
import pandas as pd
import optuna
import pickle
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# 🔹 1. 데이터 로드
file_path_train = "train_again2.csv"
file_path_test = "test_again2.csv"
sample_submission_path = "sample_submission.csv"

df_train = pd.read_csv(file_path_train)
df_test = pd.read_csv(file_path_test)
df_sample_submission = pd.read_csv(sample_submission_path)

test_ids = df_sample_submission["ID"]

# 🔹 2. Train 데이터 준비
target_col = "임신 성공 여부"
X = df_train.drop(columns=["ID", target_col], errors="ignore")
y = df_train[target_col]
X_test = df_test.drop(columns=["ID"], errors="ignore")

# 🔹 3. 범주형 변수 확인 및 결측치 처리
cat_features = X.select_dtypes(include=["object"]).columns.tolist()
print(f"📋 범주형 컬럼: {cat_features}")

# 범주형 컬럼의 결측치를 "Missing"으로 채우기
for col in cat_features:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna("Missing")
        print(f"📌 '{col}' 컬럼의 결측치를 'Missing'으로 채웠습니다.")
    if X_test[col].isnull().sum() > 0:
        X_test[col] = X_test[col].fillna("Missing")
        print(f"📌 Test 데이터의 '{col}' 컬럼 결측치를 'Missing'으로 채웠습니다.")

# 🔹 4. 클래스 가중치 설정
class_weights = {0: 0.2583, 1: 0.7417}

# 🔹 5. Optuna로 하이퍼파라미터 튜닝
def objective(trial):
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Poisson", "Bayesian"])
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
        "verbose": 0,
        "nan_mode": "Min"  # 수치형 결측치 처리
    }
    
    if bootstrap_type == "Poisson":
        params["subsample"] = trial.suggest_uniform("subsample", 0.7, 1.0)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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

# Optuna 실행
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# 최적 파라미터 저장
best_params = study.best_params
best_params["random_seed"] = 42
best_params["task_type"] = "GPU"
best_params["eval_metric"] = "AUC"
best_params["loss_function"] = "Logloss"
best_params["verbose"] = 100
best_params["nan_mode"] = "Min"

params_save_path = "cat_knn.pkl"
with open(params_save_path, "wb") as f:
    pickle.dump(best_params, f)

print(f"📁 최적화된 하이퍼파라미터가 저장되었습니다: {params_save_path}")
print(f"🎯 최적의 하이퍼파라미터: {best_params}")

# 🔹 6. 랜덤시드별 성능 평가
best_params["class_weights"] = [class_weights[0], class_weights[1]]
n_seeds = 100
seed_scores = []
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for seed in range(n_seeds):
    print(f"🌱 Seed {seed} 평가 중...")
    auc_scores = []
    
    for train_idx, valid_idx in kf.split(X, y):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        model = CatBoostClassifier(**best_params, random_seed=seed)
        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=cat_features,
            early_stopping_rounds=100,
            verbose=0
        )
        valid_preds = model.predict_proba(X_valid)[:, 1]
        auc_scores.append(roc_auc_score(y_valid, valid_preds))
    
    mean_auc = np.mean(auc_scores)
    seed_scores.append((seed, mean_auc))
    print(f"Seed {seed} 평균 AUC: {mean_auc:.6f}")

# 🔹 7. Top 5 시드 선택
seed_scores.sort(key=lambda x: x[1], reverse=True)
top_5_seeds = [seed for seed, score in seed_scores[:5]]
print(f"🏆 Top 5 Seeds: {top_5_seeds}")

# 🔹 8. Top 5 시드 + 5-Fold 앙상블
n_folds = 5
test_preds = np.zeros(len(df_test))
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

for seed in top_5_seeds:
    print(f"🔥 Seed {seed}로 K-Fold 앙상블 시작...")
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f"🔄 Fold {fold+1}/{n_folds} 학습 중 (Seed {seed})...")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostClassifier(**best_params, random_seed=seed)
        model.fit(
            X_train, y_train,
            eval_set=(X_valid, y_valid),
            cat_features=cat_features,
            early_stopping_rounds=100,
            verbose=100
        )
        test_preds += model.predict_proba(X_test)[:, 1] / (len(top_5_seeds) * n_folds)

# 🔹 9. 제출 파일 생성
df_submission = pd.DataFrame({"ID": test_ids, "probability": test_preds})
submission_file_path = "cat_knn_top5_seed_kfold_ensemble.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ Top 5 Seed + 5-Fold 앙상블 예측 결과가 '{submission_file_path}' 로 저장되었습니다.")
