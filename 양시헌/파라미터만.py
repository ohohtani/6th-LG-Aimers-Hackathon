params = {
    "iterations": trial.suggest_int("iterations", 500, 3000),
    "depth": trial.suggest_int("depth", 4, 10),
    "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0),
    "border_count": trial.suggest_int("border_count", 16, 64),
    "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Lossguide", "Depthwise"]),
    "bootstrap_type": bootstrap_type,
    "class_weights": [class_weights[0], class_weights[1]],
    "random_seed": 42,
    "task_type": "GPU",
    "eval_metric": "AUC",
    "loss_function": "Logloss",
    "verbose": 0,
    # 추가된 파라미터들
    "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),  # 리프 노드의 최소 데이터 수
    "random_strength": trial.suggest_float("random_strength", 0.1, 10.0),  # 노이즈 추가로 일반화 성능 향상
    "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 255),  # 원-핫 인코딩 적용 범주 수
    "od_type": "Iter",  # 과적합 감지기 (Iter: 반복 횟수 기반)
    "od_wait": trial.suggest_int("od_wait", 10, 50)  # 과적합 감지 대기 반복 횟수
}
