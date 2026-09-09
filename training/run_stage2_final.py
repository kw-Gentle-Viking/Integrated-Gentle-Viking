def build_stage2_config(champion: dict, today: str) -> dict:
    return {
        "train_start": "2019-01-02", "train_end": today,
        "columns": champion["columns"],
        "state_size": champion["state_size"], "attention_heads": champion["attention_heads"],
        "lstm_layers": champion["lstm_layers"], "dropout": champion["dropout"],
        "run_name": f"stage2-final-{champion['version']}",
    }
