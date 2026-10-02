# Timing probe (CPU)

code_commit=a64a590, data_version=adj1. Fixed effect = ticker mean score in val; OOT uses the val means. SE ~0.014 (val) / ~0.018 (OOT): differences under 2 SE are noise.

| scorer | window | raw IC | fixed-effect IC | timing IC | timing SE | timing top-20% minus bottom-20% /day |
|---|---|---|---|---|---|---|
| reversal (-log_ret) | val_2024 | +0.0625 | -0.0648 | +0.0684 | +0.0105 | +0.0032 |
| reversal (-log_ret) | oot_2026 | +0.0423 | -0.0081 | +0.0422 | +0.0155 | +0.0025 |
| static low-vol (-train mean vol) | val_2024 | +0.0409 | +0.0409 | n/a | n/a | n/a |
| static low-vol (-train mean vol) | oot_2026 | +0.0433 | +0.0433 | n/a | n/a | n/a |
| HGB reg (z) F3 | val_2024 | +0.0564 | -0.0535 | +0.0701 | +0.0094 | +0.0028 |
| HGB reg (z) F3 | oot_2026 | +0.0308 | -0.0183 | +0.0393 | +0.0128 | +0.0012 |
| HGB reg (date-demeaned z) F3 | val_2024 | +0.0514 | -0.0622 | +0.0662 | +0.0091 | +0.0026 |
| HGB reg (date-demeaned z) F3 | oot_2026 | +0.0408 | -0.0300 | +0.0455 | +0.0134 | +0.0023 |
| HGB clf (fixed label) F3 | val_2024 | +0.0578 | -0.0161 | +0.0646 | +0.0090 | +0.0033 |
| HGB clf (fixed label) F3 | oot_2026 | +0.0371 | +0.0131 | +0.0351 | +0.0135 | +0.0011 |

Reference (docs/signal_diagnosis.md): V3 TFT raw 0.0412/0.0392, fixed 0.0409/0.0392, timing -0.0253/-0.0393.
