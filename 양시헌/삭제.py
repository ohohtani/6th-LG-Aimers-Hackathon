import pandas as pd

# 처리할 CSV 파일 목록
file_paths = ["train3.csv", "test3.csv"]  # 대상 파일 경로 지정

# 삭제할 컬럼 목록
columns_to_remove = ["임신 시도 또는 마지막 임신 경과 연수", "시술 유형", "착상 전 유전 검사 사용 여부", 
                     "착상 전 유전 진단 사용 여부", "남성 주 불임 원인", "남성 부 불임 원인", "여성 주 불임 원인", 
                     "여성 부 불임 원인", "부부 주 불임 원인", "부부 부 불임 원인", "불임 원인 - 여성 요인",
                     "불임 원인 - 자궁경부 문제", "불임 원인 - 정자 농도", "불임 원인 - 정자 면역학적 요인", 
                     "불임 원인 - 정자 운동성", "불임 원인 - 정자 형태", "IVF 시술 횟수", "IVF 임신 횟수",
                     "DI 임신 횟수", "총 출산 횟수", "DI 출산 횟수", "미세주입된 난자 수",
                     "해동 난자 수", "저장된 신선 난자 수", "혼합된 난자 수", "정자 출처", "신선 배아 사용 여부",
                     "기증 배아 사용 여부", "대리모 여부", "PGD 시술 여부", "PGS 시술 여부",
                     "난자 채취 경과일", "난자 해동 경과일", "난자 혼합 경과일"]  # 원하는 컬럼 이름 지정

# 각 파일에 대해 반복 수행
for file_path in file_paths:
    # CSV 파일 로드
    df = pd.read_csv(file_path)
    
    # 특정 컬럼 삭제
    df.drop(columns=columns_to_remove, inplace=True, errors='ignore')  # errors='ignore'는 존재하지 않는 컬럼 무시

    # 수정된 CSV 저장 (원본 이름 + '_updated' 버전으로 저장)
    new_file_path = file_path.replace(".csv", "_updated.csv")
    df.to_csv(new_file_path, index=False)

    print(f"파일 {file_path} 처리 완료 → {new_file_path}로 저장됨")
