from data_collection.backfill_yf_fred import shift_us_date_to_kr


def test_shifts_date_forward_by_one_day():
    assert shift_us_date_to_kr("2026-09-04") == "2026-09-05"


def test_shifts_across_month_boundary():
    assert shift_us_date_to_kr("2026-08-31") == "2026-09-01"
