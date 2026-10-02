from training.config import build_tft_config


def test_config_sets_three_output_classes():
    cfg = build_tft_config(
        feature_columns={"historical": ["log_ret", "disparity_20"], "future": ["time_progress"],
                          "static_cardinalities": [21, 3]},
        num_classes=3,
    )
    assert cfg.task_type == "classification"
    assert cfg.model.num_classes == 3
    assert cfg.data_props.num_historical_numeric == 2
    assert cfg.data_props.num_future_numeric == 1
    assert cfg.data_props.static_categorical_cardinalities == [21, 3]


def test_config_encoder_state_size_defaults():
    cfg = build_tft_config(
        feature_columns={"historical": ["a"], "future": ["b"], "static_cardinalities": [21, 3]},
        num_classes=3,
    )
    assert cfg.model.state_size > 0
    assert cfg.model.attention_heads > 0
