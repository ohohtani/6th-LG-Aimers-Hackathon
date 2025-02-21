import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

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

# 🔹 6. ✅ 저장된 최적 하이퍼파라미터 로드
print("📂 저장된 XGBoost & CatBoost 하이퍼파라미터 로드 중...")
with open("best_xgb_params1.pkl", "rb") as f:
    best_xgb_params = pickle.load(f)
with open("best_cat_params1.pkl", "rb") as f:
    best_cat_params = pickle.load(f)

# 🔹 7. 최적 모델 학습
print("🚀 최적화된 XGBoost & CatBoost 모델 학습 시작...")

model_xgb = XGBClassifier(**best_xgb_params)
model_cat = CatBoostClassifier(**best_cat_params)

model_xgb.fit(X_xgb, y, verbose=100)
model_cat.fit(X, y, cat_features=cat_features, verbose=100)

print("✅ 모델 학습 완료!")

# ✅ XGBoost & CatBoost 예측값 계산
y_pred_xgb_test = model_xgb.predict_proba(X_test_xgb)[:, 1]
y_pred_cat_test = model_cat.predict_proba(df_test_cat)[:, 1]

# ✅ Soft Voting 적용 (XGBoost & CatBoost 평균)
y_pred_test_ensemble = (y_pred_xgb_test + y_pred_cat_test) / 2

print("🔄 캘리브레이션 적용 중...")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
calibrated_preds = np.zeros(len(X_test_xgb))
auc_before_calibration = []
auc_after_calibration = []
num = 1

for train_idx, valid_idx in kf.split(X, y):
    print(f"🔄 현재 {num}번째 캘리브레이션 진행 중...")

    X_train_xgb, X_valid_xgb = X_xgb.iloc[train_idx], X_xgb.iloc[valid_idx]
    X_train_cat, X_valid_cat = X_cat.iloc[train_idx], X_cat.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # 개별 모델 학습
    model_xgb = XGBClassifier(**best_xgb_params)
    model_cat = CatBoostClassifier(**best_cat_params)

    model_xgb.fit(X_train_xgb, y_train, eval_set=[(X_valid_xgb, y_valid)], verbose=0)
    model_cat.fit(X_train_cat, y_train, eval_set=(X_valid_cat, y_valid), cat_features=cat_features, verbose=0)

    # XGBoost & CatBoost 예측
    y_pred_xgb_valid = model_xgb.predict_proba(X_valid_xgb)[:, 1]
    y_pred_cat_valid = model_cat.predict_proba(X_valid_cat)[:, 1]

    # Soft Voting (XGBoost & CatBoost 평균)
    y_pred_valid_ensemble = (y_pred_xgb_valid + y_pred_cat_valid) / 2

    # 캘리브레이션 전 AUC 계산
    auc_before = roc_auc_score(y_valid, y_pred_valid_ensemble)
    auc_before_calibration.append(auc_before)

    # ✅ XGBoost & CatBoost 캘리브레이터 적용
    print("✅ 캘리브레이터 적용")
    calibrator_xgb = CalibratedClassifierCV(estimator=model_xgb, method="sigmoid", cv="prefit")
    calibrator_cat = CalibratedClassifierCV(estimator=model_cat, method="sigmoid", cv="prefit")

    calibrator_xgb.fit(X_valid_xgb, y_valid)
    calibrator_cat.fit(X_valid_cat, y_valid)

    # ✅ 캘리브레이션된 확률 예측값 출력
    calibrated_xgb_valid_preds = calibrator_xgb.predict_proba(X_valid_xgb)[:, 1]
    calibrated_cat_valid_preds = calibrator_cat.predict_proba(X_valid_cat)[:, 1]

    # ✅ 캘리브레이션 후 Soft Voting 적용
    calibrated_valid_preds = (calibrated_xgb_valid_preds + calibrated_cat_valid_preds) / 2

    # 캘리브레이션 후 AUC 계산
    auc_after = roc_auc_score(y_valid, calibrated_valid_preds)
    auc_after_calibration.append(auc_after)

    # ✅ 테스트 데이터에 대해 캘리브레이션 적용
    calibrated_xgb_test_preds = calibrator_xgb.predict_proba(X_test_xgb)[:, 1]
    calibrated_cat_test_preds = calibrator_cat.predict_proba(df_test_cat)[:, 1]

    # ✅ 캘리브레이션된 Soft Voting 적용
    calibrated_preds += (calibrated_xgb_test_preds + calibrated_cat_test_preds) / (2 * kf.n_splits)

    num += 1

# ✅ AUC 비교 출력
print(f"\n🎯 캘리브레이션 전 평균 ROC-AUC: {np.mean(auc_before_calibration):.6f}")
print(f"🔥 캘리브레이션 후 평균 ROC-AUC: {np.mean(auc_after_calibration):.6f}")

print("✅ 캘리브레이션 완료!")


# 🔹 9. 최종 제출 파일 저장 (캘리브레이션 적용된 확률 값 사용)
df_submission = pd.DataFrame({"ID": test_ids, "probability": calibrated_preds})
#submission_file_path = "final_submission_calibrated.csv"
submission_file_path = "앙상블_캘리_K_FOLD.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ 최적화된 Soft Voting 앙상블 + 캘리브레이션 완료!")
print(f"📂 최종 예측 결과 저장: {submission_file_path}")
