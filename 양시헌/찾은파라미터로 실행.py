import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# 🔹 1. 데이터 로드
file_path_train = "train_processed2.csv"
file_path_test = "test_processed2.csv"
sample_submission_path = "sample_submission.csv"

df_train = pd.read_csv(file_path_train)
df_test = pd.read_csv(file_path_test)
df_sample_submission = pd.read_csv(sample_submission_path)

# 🔹 2. 'ID' 컬럼 유지 (sample_submission을 위해 필요)
test_ids = df_sample_submission["ID"]

# 🔹 3. Train 데이터 준비
X = df_train.drop(columns=["ID", "임신 성공 여부"], errors="ignore")  # ID 제거 (없어도 에러 방지)
y = df_train["임신 성공 여부"]

# 🔹 4. Train 데이터 분할 (훈련 80%, 검증 20%)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 5. 이미 찾은 최적의 하이퍼파라미터 적용
best_params = {
    "n_estimators": 1398,
    "max_depth": 5,
    "learning_rate": 0.01190106612463671,
    "subsample": 0.49092217592801435,
    "colsample_bytree": 0.7501741111312262,
    "colsample_bylevel": 0.7688340115363569,
    "gamma": 0.002385237676448879,
    "min_child_weight": 1,
    "reg_lambda": 13.247279142964127,
    "reg_alpha": 1.4830612333216637,
    "eval_metric": "auc",
    "tree_method": "hist",  # GPU 사용
    "missing": np.nan,
    "random_state": 42,
    "use_label_encoder": False,
}

# 🔹 6. 모델 학습
model = XGBClassifier(**best_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)

# 🔹 7. 검증 데이터 평가 (ROC-AUC 스코어 계산)
valid_preds = model.predict_proba(X_valid)[:, 1]
roc_auc = roc_auc_score(y_valid, valid_preds)
print(f"Final ROC-AUC Score on Validation Set: {roc_auc:.4f}")

# 🔹 8. 테스트 데이터 예측
X_test = df_test.drop(columns=["ID"], errors="ignore")  # ID 제거 (없어도 에러 방지)
test_preds = model.predict_proba(X_test)[:, 1]

# 🔹 9. sample_submission 형식으로 변환
df_submission = pd.DataFrame({"ID": test_ids, "probability": test_preds})

# 🔹 10. 최종 CSV 파일 저장
submission_file_path = "xgboost_optuna3.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"최적화된 모델의 예측 결과가 {submission_file_path} 로 저장되었습니다.")
