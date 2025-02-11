import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# 🔹 데이터 로드
train_file_path = "train_processed2.csv"
df_train = pd.read_csv(train_file_path)

# 🔹 'ID' 컬럼 제거 (Train)
if "ID" in df_train.columns:
    df_train.drop(columns=["ID"], inplace=True)

# 🔹 Train 데이터 준비
X = df_train.drop(columns=["임신 성공 여부"])
y = df_train["임신 성공 여부"]

# 🔹 Train 데이터 분할 (훈련 80%, 검증 20%)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 기존 최적 하이퍼파라미터 (이전 튜닝 결과 기반)
best_params = {
    "n_estimators": 1460,
    "max_depth": 5,
    "learning_rate": 0.0135,
    "subsample": 0.5004,
    "colsample_bytree": 0.809,
    "colsample_bylevel": 0.8224,
    "gamma": 0.0021,
    "min_child_weight": 1,
    "reg_lambda": 13.1,
    "reg_alpha": 1.43,
}

# 🔹 Optuna 최적화 함수
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 
                                          int(best_params["n_estimators"] * 0.9), 
                                          int(best_params["n_estimators"] * 1.1)),
        "max_depth": trial.suggest_int("max_depth", 
                                       max(3, best_params["max_depth"] - 1), 
                                       best_params["max_depth"] + 1),
        "learning_rate": trial.suggest_loguniform("learning_rate", 
                                                  best_params["learning_rate"] * 0.8, 
                                                  best_params["learning_rate"] * 1.2),
        "subsample": trial.suggest_uniform("subsample", 
                                           best_params["subsample"] * 0.9, 
                                           min(1.0, best_params["subsample"] * 1.1)),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 
                                                  best_params["colsample_bytree"] * 0.9, 
                                                  min(1.0, best_params["colsample_bytree"] * 1.1)),
        "colsample_bylevel": trial.suggest_uniform("colsample_bylevel", 
                                                   best_params["colsample_bylevel"] * 0.9, 
                                                   min(1.0, best_params["colsample_bylevel"] * 1.1)),
        "gamma": trial.suggest_loguniform("gamma", 
                                          best_params["gamma"] * 0.8, 
                                          best_params["gamma"] * 1.2),
        "min_child_weight": trial.suggest_int("min_child_weight", 
                                              max(1, best_params["min_child_weight"] - 1), 
                                              best_params["min_child_weight"] + 1),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 
                                               best_params["reg_lambda"] * 0.8, 
                                               best_params["reg_lambda"] * 1.2),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 
                                              best_params["reg_alpha"] * 0.8, 
                                              best_params["reg_alpha"] * 1.2),
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42,
    }

    # 🔹 모델 학습 및 평가
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    preds = model.predict_proba(X_valid)[:, 1]
    
    return roc_auc_score(y_valid, preds)

# 🔹 Optuna 실행 (기존 값 중심으로 미세 탐색)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)  # 기존 값 기반이므로 n_trials를 줄여도 됨

# 🔹 최적 하이퍼파라미터 출력
print(f"Best ROC-AUC: {study.best_value}")
print("Best Parameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
