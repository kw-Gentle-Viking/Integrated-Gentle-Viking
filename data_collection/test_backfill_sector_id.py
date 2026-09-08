from data_collection.backfill_sector_id import match_sector_id, SECTOR_NAMES, UNCLASSIFIED_SECTOR_ID


def test_exact_match_returns_correct_index():
    assert match_sector_id("전기·전자") == SECTOR_NAMES.index("전기·전자") == 8
    assert match_sector_id("음식료·담배") == 0
    assert match_sector_id("일반서비스") == 19


def test_unmatched_name_falls_back_to_unclassified():
    # "IT 서비스"는 KIS 현재가 조회의 더 세분화된 분류라 20개 업종지수 목록에 없음(실제 라이브 검증 사례)
    assert match_sector_id("IT 서비스") == UNCLASSIFIED_SECTOR_ID == 20


def test_partial_or_case_variant_does_not_loosely_match():
    # 부분 문자열/공백 변형은 정확매칭 대상이 아님 — 오분류보다 unclassified가 안전
    assert match_sector_id("전기전자") == UNCLASSIFIED_SECTOR_ID
    assert match_sector_id(" 전기·전자 ") == UNCLASSIFIED_SECTOR_ID


def test_sector_names_list_has_20_entries_in_documented_order():
    assert len(SECTOR_NAMES) == 20
    assert SECTOR_NAMES[0] == "음식료·담배"
    assert SECTOR_NAMES[19] == "일반서비스"
