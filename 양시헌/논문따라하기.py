import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 🔹 1. 데이터 로드
file_path_train = "train3_updated.csv"
file_path_test = "test3_updated.csv"

df_train = pd.read_csv(file_path_train)
df_test = pd.read_csv(file_path_test)

# 🔹 2. 타깃 및 특징 분리
target_col = "임신 성공 여부"

X_train = df_train.drop(columns=["ID", target_col], errors="ignore")
y_train = df_train[target_col]
X_test = df_test.drop(columns=["ID"], errors="ignore")

# 🔹 3. 잘못된 값 제거 (시술 당시 나이에서 999 등 제거)
invalid_age_values = [999]
X_train = X_train[~X_train["시술 당시 나이"].isin(invalid_age_values)]

# 🔹 4. 시술 당시 나이 범주형 인코딩 (훈련 데이터 기준으로만 인코딩)
age_bins = [18, 34, 37, 39, 42, 44, 50]
age_labels = [0, 1, 2, 3, 4, 5]

def map_age(age):
    if pd.isna(age):
        return np.nan
    for i, upper_bound in enumerate(age_bins[1:], start=0):
        if age <= upper_bound:
            return age_labels[i]
    return age_labels[-1]

X_train["시술 당시 나이_범주"] = X_train["시술 당시 나이"].apply(map_age)
X_test["시술 당시 나이_범주"] = X_test["시술 당시 나이"].apply(map_age)

# 🔹 5. 결측치 처리 (훈련 데이터 기반)
train_medians = X_train.median()  # 훈련 데이터로만 median 계산
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)  # 테스트는 훈련 통계값으로만 채움

# 🔹 6. 정규화 (StandardScaler 사용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 훈련 데이터로만 fit
X_test_scaled = scaler.transform(X_test)        # 테스트 데이터는 transform만

# 🔹 7. 최종 DataFrame으로 변환
X_train_processed = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_processed = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# 🔹 8. 전처리된 데이터 확인
import ace_tools as tools; tools.display_dataframe_to_user(name="전처리된 훈련 데이터", dataframe=X_train_processed)
tools.display_dataframe_to_user(name="전처리된 테스트 데이터", dataframe=X_test_processed)

# 🔹 9. 저장 (필요 시)
X_train_processed.to_csv("train_processed.csv", index=False)
X_test_processed.to_csv("test_processed.csv", index=False)
