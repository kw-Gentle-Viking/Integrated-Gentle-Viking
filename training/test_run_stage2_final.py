from training.run_stage2_final import build_stage2_config


def test_stage2_uses_full_date_range_and_champion_columns():
    champion = {"version": "v2", "columns": ["log_ret", "est_rebalancing_flow"],
                 "state_size": 64, "attention_heads": 4, "lstm_layers": 2, "dropout": 0.15}
    cfg = build_stage2_config(champion, today="2026-09-08")
    assert cfg["train_start"] == "2019-01-02"
    assert cfg["train_end"] == "2026-09-08"
    assert cfg["columns"] == ["log_ret", "est_rebalancing_flow"]
    assert cfg["state_size"] == 64
    assert cfg["run_name"] == "stage2-final-v2"
