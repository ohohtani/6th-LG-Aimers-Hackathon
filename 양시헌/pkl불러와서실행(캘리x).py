import numpy as np
import pandas as pd
import pickle
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# 🔹 1. 데이터 로드
file_path_train = "train_again.csv"
file_path_test = "test_again.csv"
sample_submission_path = "sample_submission.csv"

df_train = pd.read_csv(file_path_train)
df_test = pd.read_csv(file_path_test)
df_sample_submission = pd.read_csv(sample_submission_path)

# 🔹 2. 'ID' 컬럼 유지
test_ids = df_sample_submission["ID"]

# 🔹 3. 데이터 준비 (결측치 처리는 CatBoost에 맡김)
target_col = "임신 성공 여부"
if target_col not in df_train.columns:
    raise ValueError(f"❌ '{target_col}' 컬럼이 존재하지 않습니다. 데이터 확인 필요!")

X = df_train.drop(columns=["ID", target_col], errors="ignore")
y = df_train[target_col]
X_test = df_test.drop(columns=["ID"], errors="ignore")

# 파생 변수 생성 함수
def add_derived_features(df):
    df = df.copy()
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
    return df

X = add_derived_features(X)
X_test = add_derived_features(X_test)

# 🔹 4. 범주형 변수 확인
cat_features = X.select_dtypes(include=["object"]).columns.tolist()

# 🔹 5. 클래스 가중치 설정
class_weights = {0: 0.2583, 1: 0.7417}

# 🔹 6. 저장된 최적 파라미터 불러오기
params_save_path = "cat_after.pkl"
with open(params_save_path, "rb") as f:
    best_params = pickle.load(f)

print(f"📁 저장된 하이퍼파라미터를 불러왔습니다: {params_save_path}")
print(f"🎯 불러온 하이퍼파라미터: {best_params}")

# 기본 설정 추가
best_params["task_type"] = "GPU"  # GPU 없으면 "CPU"로 변경
best_params["eval_metric"] = "AUC"
best_params["loss_function"] = "Logloss"
best_params["verbose"] = 0  # 탐색 중 출력 최소화
best_params["class_weights"] = [class_weights[0], class_weights[1]]

# 🔹 7. 검증 세트 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🔹 8. 최적 random_seed 탐색
seed_range = range(0, 100)  # 0부터 99까지 탐색 (시간 고려, 필요 시 범위 조정)
val_scores = []

for seed in seed_range:
    best_params["random_seed"] = seed
    model = CatBoostClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features, 
              early_stopping_rounds=100, verbose=0)
    val_preds = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, val_preds)
    val_scores.append((seed, score))
    print(f"Seed {seed} - ROC-AUC: {score:.5f}")

# 상위 3개 시드 선택
top_seeds = sorted(val_scores, key=lambda x: x[1], reverse=True)[:3]
top_seeds = [s[0] for s in top_seeds]
print(f"선택된 상위 3개 시드: {top_seeds}")

# 🔹 9. 상위 시드로 앙상블
test_preds_optimal = np.zeros(len(X_test))
best_params["verbose"] = 100  # 학습 과정 출력

for seed in top_seeds:
    best_params["random_seed"] = seed
    model = CatBoostClassifier(**best_params)
    model.fit(X, y, cat_features=cat_features, verbose=100)  # 전체 데이터로 학습
    test_preds_optimal += model.predict_proba(X_test)[:, 1] / len(top_seeds)

# 🔹 10. 제출 파일 생성
df_submission = pd.DataFrame({"ID": test_ids, "probability": test_preds_optimal})
submission_file_path = "cat_after_ensemble_optimal.csv"
df_submission.to_csv(submission_file_path, index=False)

print(f"✅ CatBoost 최적 시드 앙상블 예측 결과가 '{submission_file_path}' 로 저장되었습니다.")
