import optuna


def run_optuna_study(objective_fn, n_trials: int = 20) -> optuna.Study:
    """objective_fn(trial) -> macro_f1 (maximize). trial에서 state_size, attention_heads,
    lstm_layers, dropout, lr을 suggest해서 training.train.run_training에 넘기는 방식으로 구성."""
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_fn, n_trials=n_trials)
    return study
