# Chronos-2 timing probe

One `predict_df` call per row (close + volume_log1p, context 512d, prediction_length=1).

| device | n_tickers (of universe) | model load (s) | infer (s) | ms/ticker | extrapolated full-universe/day (s) |
|---|---|---|---|---|---|
| cpu | 25/200 | 0.96 | 1.082 | 43.3 | 8.7 |
| cpu | 200/200 | 1.07 | 11.449 | 57.2 | 11.4 |
| cuda | 200/200 | 1.44 | 0.818 | 4.1 | 0.8 |
