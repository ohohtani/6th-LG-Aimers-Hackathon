import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1️⃣ Train 데이터 불러오기 
file_path_train = "train_processed.csv"
df_train = pd.read_csv(file_path_train)

# 🔹 'ID' 컬럼 제거 (Train)
if "ID" in df_train.columns:
    df_train.drop(columns=["ID"], inplace=True)

# 6️⃣ Train 데이터 준비
X = df_train.drop(columns=["임신 성공 여부"])
y = df_train["임신 성공 여부"]

# 7️⃣ Train 데이터 분할 (훈련 80%, 검증 20%)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 9️⃣ Test 데이터 불러오기 (전처리 완료된 데이터)
file_path_test = "test_processed.csv"
df_test = pd.read_csv(file_path_test)

# 🔹 'ID' 컬럼 제거 (Test)
if "ID" in df_test.columns:
    df_test.drop(columns=["ID"], inplace=True)

# 🔟 XGBoost 모델 생성 (논문 기반 최적화 적용)
model = XGBClassifier(
    n_estimators=1000,   # 트리 개수 증가 (Early Stopping으로 조정)
    max_depth=5,         # 트리 깊이 조정
    learning_rate=0.03,  # Learning Rate 감소 (Shrinkage)
    subsample=0.8,       # 데이터 샘플링 비율 (Bagging 효과)
    colsample_bytree=0.7, # Feature Sampling 비율 (Column Subsampling)
    colsample_bylevel=0.7,# 레벨 단위 Feature Sampling 비율
    gamma=1.5,           # Split 여부를 결정하는 Min Loss Reduction
    min_child_weight=3,  # 과적합 방지 (Leaf 노드가 최소 가져야 하는 가중치)
    reg_lambda=10,       # L2 Regularization (Ridge)
    reg_alpha=0.5,       # L1 Regularization (Lasso)
    eval_metric="auc",  
    missing=np.nan,      # 결측값 자동 처리
    random_state=42,
    use_label_encoder=False,
    early_stopping_rounds=30  # 30번 연속 개선 없으면 종료
)

# 🔟 모델 학습 (가중치 적용하여 조정된 Feature 반영)
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=100
)

# 🔟 검증 데이터 평가 (ROC-AUC)
y_valid_pred = model.predict_proba(X_valid)[:, 1]  
roc_auc = roc_auc_score(y_valid, y_valid_pred)
print(f"🎯 XGBoost 검증 데이터 ROC-AUC: {roc_auc:.4f}")

# 1️⃣1️⃣ Test 데이터 예측 (확률값 저장)
y_pred_test_proba = model.predict_proba(df_test)[:, 1]
y_pred_test_proba = np.round(y_pred_test_proba, 5)

# 1️⃣2️⃣ 제출 파일 저장
file_path_submission = "sample_submission.csv"
df_submission = pd.read_csv(file_path_submission)
df_submission["probability"] = y_pred_test_proba
file_path_final_submission = "xgboost_submission_optimized.csv"
df_submission.to_csv(file_path_final_submission, index=False)

print("✅ XGBoost 모델 학습 및 예측 완료!")
print(f"🎯 최종 ROC-AUC 점수: {roc_auc:.4f}")
print(f"📂 '{file_path_final_submission}' 파일이 저장되었습니다.")
