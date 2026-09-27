"""임베딩 배열 + (ticker,date) 키를 표 피처 DataFrame과 내부조인. 커버리지가 기준 미달이면
예외 -- 조용한 드롭 금지(spec §에러 처리)."""
import numpy as np
import pandas as pd


def join_embeddings_with_tabular(keys: list[tuple[str, str]], emb: np.ndarray, tab_df: pd.DataFrame,
                                 ticker_col: str = "ticker", date_col: str = "date_s") -> pd.DataFrame:
    if len(set(keys)) != len(keys):
        raise ValueError(f"join_embeddings_with_tabular: duplicate embedding keys ({len(keys)} keys, "
                         f"{len(set(keys))} unique)")
    tab_keys = list(zip(tab_df[ticker_col], tab_df[date_col]))
    if len(set(tab_keys)) != len(tab_keys):
        raise ValueError(f"join_embeddings_with_tabular: duplicate tabular keys ({len(tab_keys)} rows, "
                         f"{len(set(tab_keys))} unique)")
    emb_df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(emb.shape[1])])
    emb_df[ticker_col] = [k[0] for k in keys]
    emb_df[date_col] = [k[1] for k in keys]
    return emb_df.merge(tab_df, on=[ticker_col, date_col], how="inner")


def assert_join_coverage(n_joined: int, n_emb: int, n_tab: int, min_ratio: float, label: str) -> None:
    baseline = min(n_emb, n_tab)
    ratio = (n_joined / baseline) if baseline else 1.0
    if ratio < min_ratio:
        raise ValueError(
            f"[{label}] join coverage {ratio:.3f} below min_ratio={min_ratio} "
            f"(joined={n_joined}, n_emb={n_emb}, n_tab={n_tab})")
