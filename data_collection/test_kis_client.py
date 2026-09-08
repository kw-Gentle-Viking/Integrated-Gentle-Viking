from datetime import datetime
import pytest
from data_collection.kis_client import assert_outside_production_window


def test_raises_during_realtime_collection_window():
    weekday_0955 = datetime(2026, 9, 9, 9, 55)  # 수요일
    with pytest.raises(RuntimeError, match="production collection window"):
        assert_outside_production_window(weekday_0955)


def test_raises_during_kis_collector_window():
    weekday_1555 = datetime(2026, 9, 9, 15, 55)
    with pytest.raises(RuntimeError, match="production collection window"):
        assert_outside_production_window(weekday_1555)


def test_raises_during_batch_collector_window():
    weekday_1605 = datetime(2026, 9, 9, 16, 5)
    with pytest.raises(RuntimeError, match="production collection window"):
        assert_outside_production_window(weekday_1605)


def test_allows_evening_window():
    weekday_2000 = datetime(2026, 9, 9, 20, 0)
    assert_outside_production_window(weekday_2000)  # no raise


def test_allows_weekend():
    saturday_1000 = datetime(2026, 9, 12, 10, 0)
    assert_outside_production_window(saturday_1000)  # no raise
