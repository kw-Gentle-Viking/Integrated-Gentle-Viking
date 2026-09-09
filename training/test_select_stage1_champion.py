from training.select_stage1_champion import select_champion


def test_selects_highest_val_macro_f1():
    versions = [
        {"version": "v1", "macro_f1_val": 0.41, "columns": ["a", "b"]},
        {"version": "v2", "macro_f1_val": 0.53, "columns": ["a", "b", "c"]},
        {"version": "v3", "macro_f1_val": 0.47, "columns": ["a"]},
    ]
    champion = select_champion(versions)
    assert champion["version"] == "v2"


def test_raises_on_empty_versions():
    import pytest
    with pytest.raises(ValueError, match="no versions"):
        select_champion([])
