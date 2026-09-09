def select_champion(versions: list[dict]) -> dict:
    if not versions:
        raise ValueError("no versions to select from")
    return max(versions, key=lambda v: v["macro_f1_val"])
