import pytest

from training.run_stage2_final_live import parse_args


def test_weight_scheme_defaults_to_balanced_and_promote_flag_off():
    args = parse_args([])
    assert args.weight_scheme == "balanced"
    assert args.promote_to_serving is False
    assert args.epochs == 15


def test_weight_scheme_accepts_the_three_schemes_and_rejects_others():
    for scheme in ("balanced", "uniform", "mild"):
        assert parse_args(["--weight-scheme", scheme]).weight_scheme == scheme
    with pytest.raises(SystemExit):
        parse_args(["--weight-scheme", "bogus"])


def test_promote_flag_is_explicit_opt_in():
    assert parse_args(["--weight-scheme", "mild", "--promote-to-serving"]).promote_to_serving is True
