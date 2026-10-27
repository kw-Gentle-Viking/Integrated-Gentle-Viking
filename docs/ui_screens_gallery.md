# 화면 캡처 모음 (전체 페이지, 2026-10-27)

발표 자료용 참고 — 프론트엔드에 실제로 존재하는 화면을 전부 캡처했다. 로그인 없이, DB에서 새
발급한 토큰을 브라우저에 주입해서 접근했다(실제 비밀번호 사용 안 함). 메인 발표 문서
(`capstone_overview_for_presentation.md`)에는 이 중 포트폴리오 화면 하나만 골라 넣었고,
나머지는 필요한 걸 여기서 골라 쓰면 된다.

| 파일 | 화면 | 비고 |
|---|---|---|
| `images/screens/01_landing.png` | 랜딩 페이지 | 비로그인, 일부 목업 데이터 |
| `images/screens/02_login.png` | 로그인 | |
| `images/screens/03_signup.png` | 회원가입 (1/2단계) | |
| `images/screens/04_dashboard.png` | 홈(대시보드) | 실제 KIS 모의계좌 잔고(10,000,000원), 실시간 코스피/코스닥/환율, 실시간 종목 랭킹 |
| `images/screens/05_market.png` | 실시간 종목 랭킹 | 실제 KIS 거래대금 순위, 삼성전자(바구니 담김) 하이라이트 확인됨 |
| `images/screens/06_market_detail.png` | 종목 상세(005930) | 실제 캔들차트, 매수/매도 버튼. 제목이 종목코드만 나오고 종목명(삼성전자)은 안 뜸 — 안 고침 |
| `images/screens/07_portfolio.png` | 내 주식보기(포트폴리오+자동매매) | 이번에 이름/시세 버그 고친 뒤 버전. 메인 문서에 들어간 것과 동일 |
| `images/screens/08_watchlist.png` | 관심 종목 | 빈 상태(아직 관심종목 없음) |
| `images/screens/09_ai_report.png` | AI 리포트 홈 | 캡처 시점에 Gemini 리포트 생성 중 상태로 찍힘 |
| `images/screens/10_ai_report_predict.png` | 종목별 AI 예측 상세(005930) | 매수 33%/관망 25%/매도 42%, 모델 버전(`staticwd`)까지 실제 값. 종목명 버그 고친 뒤 버전("삼성전자"로 정상 표시). 하단 "추론 결과 분석 리포트"는 여전히 전부 "백엔드 응답에 상세 분석 리포트가 포함되지 않았습니다" — 별도 확인 필요 |
| `images/screens/11_mypage.png` | 마이페이지 - 자산 | 실제 KIS 모의계좌 잔고·주문내역 구조 |
| `images/screens/12_notifications.png` | 알림 | 목업 데이터로 보임 |
| `images/screens/13_settings.png` | 설정 | |

## 이번에 같이 고친 것

- `Front-Gentle-Viking` 사이드바에서 "내 주식보기" 메뉴 글자가 두 줄로 밀리던 문제 —
  `whitespace-nowrap` 추가로 해결.
- 포트폴리오 화면 종목명/시세 버그 — 별도로 이미 고치고 보고함 (`fix(portfolio): ...` 커밋).
- AI 예측 상세 페이지(10번)가 "종목명 없음"으로 뜨던 버그 — 포트폴리오에서 쓴 이름 해석 함수
  (`resolveStockName`)를 공유로 빼서 이 페이지에도 적용 (`fix(ai-report): ...` 커밋).

## 발견했지만 아직 안 고친 것

- AI 예측 상세 페이지의 "추론 결과 분석 리포트" 하위 4개 항목이 전부 빈 문구 — 백엔드가 상세
  리포트를 안 채워주는 것으로 보이는데, 원인은 더 확인 안 함.
- 종목 상세 페이지(06번)가 종목코드만 보여주고 한글 종목명은 안 보여줌.

필요하면 이어서 고칠게.
