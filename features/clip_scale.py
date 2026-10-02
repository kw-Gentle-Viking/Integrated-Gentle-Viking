import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_clip_bounds(train_values: np.ndarray, lower_q: float = 0.005,
                     upper_q: float = 0.995) -> tuple[float, float]:
    clean = train_values[~np.isnan(train_values)]
    return float(np.quantile(clean, lower_q)), float(np.quantile(clean, upper_q))


def apply_clip(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    return np.clip(values, bounds[0], bounds[1])


def fit_scaler(train_df: pd.DataFrame, columns: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[columns])
    return scaler


def save_artifacts(bounds: dict, scaler: StandardScaler, path: str) -> None:
    with open(f"{path}.bounds.json", "w") as f:
        json.dump(bounds, f)
    with open(f"{path}.scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


def load_artifacts(path: str) -> tuple[dict, StandardScaler]:
    with open(f"{path}.bounds.json") as f:
        bounds = json.load(f)
    with open(f"{path}.scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return bounds, scaler
