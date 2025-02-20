# 📦 라이브러리 설치 및 불러오기
import numpy as np
import pandas as pd
import optuna
import joblib
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

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
print(f"✅ 범주형 변수 목록: {cat_features}")

# ✅ XGBoost는 범주형 변수 처리 불가 → Label Encoding 적용
if len(cat_features) > 0:
    print("🔔 XGBoost 처리용 Label Encoding 적용 중...")
    le = LabelEncoder()
    for col in cat_features:
        X[col] = le.fit_transform(X[col])
        df_test[col] = le.transform(df_test[col])
else:
    print("✅ 범주형 변수 없음. XGBoost 별도 처리 불필요.")

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

# 🔹 6. 최적 모델 불러오기
cb_model = joblib.load("saved_cb_model.pkl")
xgb_model = joblib.load("saved_xgb_model.pkl")

# 🔹 7. 소프트 보팅 확률 예측 (가중치 적용: CatBoost=7, XGBoost=3)
y_pred_cb_proba = cb_model.predict_proba(X_calib)[:, 1]
y_pred_xgb_proba = xgb_model.predict_proba(X_numeric_calib)[:, 1]

# ✅ 가중치 적용 소프트 보팅
cat_weight, xgb_weight = 7, 3
total_weight = cat_weight + xgb_weight
y_pred_proba = (cat_weight * y_pred_cb_proba + xgb_weight * y_pred_xgb_proba) / total_weight
y_pred = (y_pred_proba >= 0.5).astype(int)

# 🔹 8. 성능 평가
auc_score = roc_auc_score(y_calib, y_pred_proba)
accuracy = accuracy_score(y_calib, y_pred)
print(f"✅ 가중치 적용 Soft Voting AUC: {auc_score:.4f}")
print(f"✅ 가중치 적용 Soft Voting Accuracy: {accuracy:.4f}")

# 🔹 9. 테스트 데이터 예측 및 제출
_y_pred_cb_proba = cb_model.predict_proba(df_test)[:, 1]
_y_pred_xgb_proba = xgb_model.predict_proba(X_test_numeric)[:, 1]

ensemble_pred_proba = (cat_weight * _y_pred_cb_proba + xgb_weight * _y_pred_xgb_proba) / total_weight

df_submission = pd.DataFrame({"ID": test_ids, "probability": ensemble_pred_proba})
submission_file_path = "weighted_soft_voting_submission.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ 가중치 적용 확률 기반 제출 파일이 '{submission_file_path}'로 저장되었습니다.")
