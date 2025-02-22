import numpy as np
import pandas as pd

# --- 데이터 로드 ---
file_path_train = "train3_updated.csv"
df_train = pd.read_csv(file_path_train)

# --- 타겟 및 피처 분리 (ID와 타겟은 유지) ---
target_col = "임신 성공 여부"
X = df_train.drop(columns=[target_col], errors="ignore")  # ID는 유지
y = df_train[target_col]

# --- 전처리 함수 ---
def preprocess_data(X):
    X_new = X.copy()
    age_map = {
        "만18-34세": 26, "만35-37세": 36, "만38-39세": 38.5, "만40-42세": 41,
        "만43-44세": 43.5, "만45-50세": 47.5, "만21-25세": 23, "만26-30세": 28
    }
    X_new["시술 당시 나이"] = X_new["시술 당시 나이"].map(age_map)
    X_new["난자 기증자 나이"] = X_new["난자 기증자 나이"].replace("알 수 없음", np.nan).map(age_map, na_action="ignore")
    X_new["정자 기증자 나이"] = X_new["정자 기증자 나이"].replace("알 수 없음", np.nan).map(age_map, na_action="ignore")

    count_cols = ["총 시술 횟수", "클리닉 내 총 시술 횟수", "DI 시술 횟수", "총 임신 횟수", "IVF 출산 횟수"]
    for col in count_cols:
        X_new[col] = X_new[col].str.replace("회", "").replace({"6회 이상": "6", "6 이상": "6"})
        X_new[col] = pd.to_numeric(X_new[col], errors="coerce").fillna(0)

    X_new.fillna({"난자 기증자 나이": X_new["시술 당시 나이"], "정자 기증자 나이": X_new["시술 당시 나이"]}, inplace=True)
    return X_new

# --- 요청된 피처 추가 함수 ---
def add_selected_features(X):
    X_new = preprocess_data(X)

    # 요청된 3개 피처 추가
    X_new["embryo_age_interaction"] = X_new["이식된 배아 수"] * X_new["시술 당시 나이"]
    X_new["transfer_day_log"] = np.log1p(X_new["배아 이식 경과일"])
    X_new["fresh_embryo_product"] = X_new["수집된 신선 난자 수"] * X_new["총 생성 배아 수"]

    return X_new

# --- 데이터에 피처 추가 ---
X_train_with_features = add_selected_features(X)

# --- 타겟 다시 결합 ---
df_train_processed = X_train_with_features.copy()
df_train_processed[target_col] = y

# --- 결과 확인 ---
print("📋 전처리 및 피처 추가된 컬럼:")
print(list(df_train_processed.columns))
print("\n📊 데이터 샘플 (상위 5행):")
print(df_train_processed.head())

# --- 전처리된 데이터 저장 (선택 사항) ---
output_file = "train3_with_selected_features.csv"
df_train_processed.to_csv(output_file, index=False)
print(f"✅ 전처리된 데이터 저장: {output_file}")