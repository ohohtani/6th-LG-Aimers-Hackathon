import numpy as np
import pandas as pd

# --- 데이터 로드 ---
file_path_train = "/content/train.csv"
file_path_test = "/content/test.csv"

df_train = pd.read_csv(file_path_train)
df_test = pd.read_csv(file_path_test)

# --- 타겟 분리 (train만) ---
target_col = "임신 성공 여부"
X_train = df_train.drop(columns=[target_col], errors="ignore")
y_train = df_train[target_col]
X_test = df_test.copy()

# --- 결측치를 최빈값으로 채우는 함수 ---
def fill_missing_with_mode(df):
    df_filled = df.copy()
    for column in df_filled.columns:
        if df_filled[column].isnull().any():
            mode_value = df_filled[column].mode()[0]  # 최빈값 중 첫 번째 값 사용
            df_filled[column].fillna(mode_value, inplace=True)
            print(f"📌 '{column}' 결측치 최빈값 '{mode_value}'로 채움")
    return df_filled

# --- 전처리 함수 (이전 코드 재사용) ---
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
        if col in X_new.columns:
            X_new[col] = X_new[col].str.replace("회", "").replace({"6회 이상": "6", "6 이상": "6"})
            X_new[col] = pd.to_numeric(X_new[col], errors="coerce").fillna(0)

    X_new.fillna({"난자 기증자 나이": X_new["시술 당시 나이"], "정자 기증자 나이": X_new["시술 당시 나이"]}, inplace=True)
    return X_new

# --- 요청된 피처 추가 함수 (이전 코드 재사용) ---
def add_selected_features(X):
    X_new = preprocess_data(X)
    if "이식된 배아 수" in X_new.columns and "시술 당시 나이" in X_new.columns:
        X_new["embryo_age_interaction"] = X_new["이식된 배아 수"] * X_new["시술 당시 나이"]
    if "배아 이식 경과일" in X_new.columns:
        X_new["transfer_day_log"] = np.log1p(X_new["배아 이식 경과일"])
    if "수집된 신선 난자 수" in X_new.columns and "총 생성 배아 수" in X_new.columns:
        X_new["fresh_embryo_product"] = X_new["수집된 신선 난자 수"] * X_new["총 생성 배아 수"]
    return X_new

# --- Train 데이터 결측치 채우기 및 전처리 ---
X_train_filled = fill_missing_with_mode(X_train)
X_train_processed = add_selected_features(X_train_filled)

# 타겟 다시 결합
df_train_processed = X_train_processed.copy()
df_train_processed[target_col] = y_train

# --- Test 데이터 결측치 채우기 및 전처리 ---
X_test_filled = fill_missing_with_mode(X_test)
X_test_processed = add_selected_features(X_test_filled)

# --- 결과 확인 ---
print("\n📋 Train 데이터 - 전처리 및 피처 추가된 컬럼:")
print(list(df_train_processed.columns))
print("\n📊 Train 데이터 샘플 (상위 5행):")
print(df_train_processed.head())

print("\n📋 Test 데이터 - 전처리 및 피처 추가된 컬럼:")
print(list(X_test_processed.columns))
print("\n📊 Test 데이터 샘플 (상위 5행):")
print(X_test_processed.head())

# --- 전처리된 데이터 저장 ---
train_output_file = "train3_with_mode_filled.csv"
test_output_file = "test3_with_mode_filled.csv"

df_train_processed.to_csv(train_output_file, index=False)
X_test_processed.to_csv(test_output_file, index=False)

print(f"✅ 전처리된 Train 데이터 저장: {train_output_file}")
print(f"✅ 전처리된 Test 데이터 저장: {test_output_file}")
