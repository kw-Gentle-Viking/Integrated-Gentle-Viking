"""추론·수집 부하 상한. 종목이 많아지면 5분 추론이 밀리고 KIS 실시간 수집 호출이 타임아웃 난다
(팀 합의: 5종목). 등록(api_server START)과 추론(inference_pipeline) 양쪽에서 강제한다."""

MAX_ACTIVE_TICKERS = 5
