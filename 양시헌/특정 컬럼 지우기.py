import pandas as pd

# 1️⃣ Train 데이터 불러오기
file_path_train = "train_processed_high.csv"
df_train = pd.read_csv(file_path_train)

# 2️⃣ Test 데이터 불러오기
file_path_test = "test_processed_high.csv"
df_test = pd.read_csv(file_path_test)

# 삭제할 컬럼 목록
columns_to_drop = ["착상 전 유전 검사 사용 여부"]

# 🔹 Train 데이터에서 컬럼 삭제
df_train.drop(columns=[col for col in columns_to_drop if col in df_train.columns], inplace=True)

# 🔹 Test 데이터에서 컬럼 삭제
df_test.drop(columns=[col for col in columns_to_drop if col in df_test.columns], inplace=True)

# 4️⃣ 새로운 CSV 파일로 저장
file_path_train_new = "train_processed2.csv"
file_path_test_new = "test_processed2.csv"

df_train.to_csv(file_path_train_new, index=False)
df_test.to_csv(file_path_test_new, index=False)

print(f"✅ '시술 유형'과 '총 시술 횟수' 컬럼 삭제 완료!")
print(f"📂 새로운 Train 파일 저장: {file_path_train_new}")
print(f"📂 새로운 Test 파일 저장: {file_path_test_new}")
