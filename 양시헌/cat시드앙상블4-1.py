import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# 🔹 1. 데이터 로드
file_path_train = "train_again2.csv"
file_path_test = "test_again2.csv"
sample_submission_path = "sample_submission.csv"
params_save_path = "cat_knn.pkl"  # 저장된 파라미터 경로

# 🔹 2. 데이터 불러오기
df_train = pd.read_csv(file_path_train)
df_test = pd.read_csv(file_path_test)
df_sample_submission = pd.read_csv(sample_submission_path)

test_ids = df_sample_submission["ID"]

# 🔹 3. Train 데이터 준비
target_col = "임신 성공 여부"
X = df_train.drop(columns=["ID", target_col], errors="ignore")
y = df_train[target_col]
X_test = df_test.drop(columns=["ID"], errors="ignore")

# 🔹 4. 범주형 및 수치형 변수 확인
cat_features = X.select_dtypes(include=["object"]).columns.tolist()
num_features = X.select_dtypes(include=[np.float64, np.int64]).columns.tolist()
print(f"📋 범주형 컬럼: {cat_features}")
print(f"📋 수치형 컬럼: {num_features}")

# 🔹 5. 범주형 결측치 처리
for col in cat_features:
    X[col] = X[col].fillna("Missing")
    X_test[col] = X_test[col].fillna("Missing")

# 🔹 6. 클러스터링 컬럼 추가 확인 및 처리
if "Cluster" in X.columns:
    cat_features.append("Cluster")

# 🔹 7. 저장된 최적 파라미터 불러오기
try:
    with open(params_save_path, "rb") as f:
        best_params = pickle.load(f)
    print(f"✅ 저장된 최적 파라미터 불러오기 완료: {best_params}")
except FileNotFoundError:
    raise FileNotFoundError(f"🚫 '{params_save_path}' 파일을 찾을 수 없습니다. Optuna 튜닝을 먼저 실행하세요.")

# 🔹 8. random_seed 중복 제거
best_params.pop("random_seed", None)

# 🔹 9. 랜덤시드별 성능 평가
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

# 🔹 10. Top 5 시드 선택
seed_scores.sort(key=lambda x: x[1], reverse=True)
top_5_seeds = [seed for seed, _ in seed_scores[:5]]
print(f"🏆 Top 5 Seeds: {top_5_seeds}")

# 🔹 11. Top 5 시드 + 5-Fold 앙상블 예측
test_preds = np.zeros(len(df_test))
n_folds = 5

for seed in top_5_seeds:
    print(f"🔥 Seed {seed}로 K-Fold 앙상블 시작...")
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        print(f"🔄 Fold {fold + 1}/{n_folds} 학습 중 (Seed {seed})...")
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

# 🔹 12. 제출 파일 생성
df_submission = pd.DataFrame({"ID": test_ids, "probability": test_preds})
submission_file_path = "cat_knn_top5_seed_kfold_ensemble.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ Top 5 Seed + 5-Fold 앙상블 예측 결과가 '{submission_file_path}' 로 저장되었습니다.")
