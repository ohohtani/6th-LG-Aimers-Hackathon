import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


file_path_train = "/content/train_processed.csv"
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
file_path_test = "/content/test_processed.csv"
df_test = pd.read_csv(file_path_test)

# 🔹 'ID' 컬럼 제거 (Test)
if "ID" in df_test.columns:
    df_test.drop(columns=["ID"], inplace=True)

def objective(trial):
    # 튜닝할 하이퍼파라미터 범위 설정
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_uniform("colsample_bylevel", 0.5, 1.0),
        "gamma": trial.suggest_loguniform("gamma", 1e-3, 10.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-2, 100.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-2, 100.0),
        "eval_metric": "auc",
        "tree_method": "hist",      # GPU 사용을 위해 "hist" 사용 (gpu_hist는 deprecated)
        "device": "cuda",           # GPU를 사용하도록 지정
        "missing": np.nan,
        "random_state": 42,
        "use_label_encoder": False,
        "early_stopping_rounds": 30,
    }

    # XGBClassifier 생성 후 학습 (verbose=False로 학습 진행상황 생략)
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )
    # 검증 데이터에 대한 예측 확률 계산 (양성 클래스)
    preds = model.predict_proba(X_valid)[:, 1]
    # ROC-AUC 스코어 계산 (높을수록 좋으므로 maximize 방향으로 최적화)
    score = roc_auc_score(y_valid, preds)
    return score

# Optuna 스터디 생성 (목표: ROC-AUC 최대화)
study = optuna.create_study(direction="maximize")
# 예시: 50번의 trial로 튜닝 (n_trials는 필요에 따라 조정)
study.optimize(objective, n_trials=50)

print("Best trial:")
best_trial = study.best_trial
print(f"  ROC-AUC: {best_trial.value:.4f}")
print("  Best hyperparameters:")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")
"""
위 코드로 얻은 하이퍼 파라미터 입니다.
약 380번 (한시간,코랩 gpu T4)정도 돌렸습니다. 시헌님이 올려주신 코드에서 따로 처리 안한 상태에서 돌리면 validation 0.74005 나옵니다.
Best trial:
  ROC-AUC: 0.7401
  Best hyperparameters:
    n_estimators: 1460
    max_depth: 5
    learning_rate: 0.013528250043705115
    subsample: 0.5004290707020331
    colsample_bytree: 0.8090419117285462
    colsample_bylevel: 0.8224549845420901
    gamma: 0.0021286966552009406
    min_child_weight: 1
    reg_lambda: 13.10737878511581
    reg_alpha: 1.4380716740432997

"""
