import numpy as np
import pandas as pd
import optuna
import pickle
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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

# 🔹 4. 범주형 변수 확인 (CatBoost에서 직접 사용 가능)
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# 🔹 5. 최적의 CatBoost 하이퍼파라미터 불러오기
params_save_path = "cat_after.pkl"
with open(params_save_path, "rb") as f:
    best_params = pickle.load(f)

best_params["class_weights"] = [0.2583, 0.7417]  # 기존 가중치 유지
best_params["verbose"] = 100  # 학습 로그 표시

# 🔹 6. K-Fold 기반 최종 학습 및 예측 (앙상블 적용)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # K=5
test_preds = np.zeros(len(df_test))  # 테스트 예측값 저장

print("🔄 K-Fold 기반 최종 학습 시작...")

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    print(f"🔄 Fold {fold + 1} 학습 중...")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # CatBoost 모델 학습
    model = CatBoostClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        cat_features=cat_features,
        early_stopping_rounds=100,
        verbose=100
    )

    # 검증 데이터 예측
    valid_preds = model.predict_proba(X_valid)[:, 1]
    fold_auc = roc_auc_score(y_valid, valid_preds)
    print(f"🎯 Fold {fold + 1} ROC-AUC: {fold_auc:.8f}")

    # 테스트 데이터 예측 (각 Fold 모델의 예측값을 누적하여 평균)
    test_preds += model.predict_proba(df_test.drop(columns=["ID"], errors="ignore"))[:, 1] / kf.n_splits

# 🔹 7. 캘리브레이션 후처리 적용
print("🔄 캘리브레이션 적용 중...")

calibrated_preds = np.zeros(len(df_test))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # CatBoost 모델 재학습
    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, cat_features=cat_features, verbose=0)

    # 캘리브레이터 적용
    calibrator = CalibratedClassifierCV(base_estimator=model, method='sigmoid', cv='prefit')
    calibrator.fit(X_valid, y_valid)

    # 테스트 데이터 캘리브레이션 적용
    calibrated_preds += calibrator.predict_proba(df_test.drop(columns=["ID"], errors="ignore"))[:, 1] / kf.n_splits

print("✅ 캘리브레이션 완료!")

# 🔹 8. sample_submission 형식으로 변환
df_submission = pd.DataFrame({"ID": test_ids, "probability": calibrated_preds})

# 🔹 9. 최종 CSV 파일 저장
submission_file_path = "catboost_kfold_final_calibrated.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ K-Fold 기반 최종 CatBoost 모델의 예측 결과가 '{submission_file_path}' 로 저장되었습니다.")
