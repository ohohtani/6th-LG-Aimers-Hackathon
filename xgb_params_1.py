"""
scale_pos_weight = negative_ct/positive_ct 로 scale_pos_weight만 두고 했을때 그렇지 않았을때에 비해서 약 0.0004점 올랐음.
"""

# scale_pos_weight 값 계산
positive_ct = ((y_train == 1).sum())
negative_ct = ((y_train == 0).sum())
scale_pos_weight = negative_ct/positive_ct
               

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
        "scale_pos_weight": scale_pos_weight
    }