"""VI (변동성완화장치, Volatility Interruption) event backfill via KIS TR FHPST01390000.

Background (see docs/data_units.md 2026-09-09 entry for full research trail):
- Task 8 tried KIS TR `FHKST02900200` (endpoint `/uapi/domestic-stock/v1/quotations/inquire-vi`)
  and got a live 404 — that endpoint does not exist in the current KIS API.
- KRX's authenticated Open API (openapi.krx.co.kr, the service `KRX_API_KEY` registers against)
  was checked live on 2026-09-09: its full ~35-service catalogue (지수/주식/증권상품/채권/
  파생상품/일반상품/ESG) has no VI/circuit-breaker service at all.
- KRX's public MDC statistics portal (data.krx.co.kr) has a genuine "변동성완화장치 발동종목현황"
  page (MDCSTAT224, backing bld `dbms/MDC/STAT/issue/MDCSTAT22401`) with exactly the right
  columns, but as of 2026-09-09 every call to its JSON data endpoint
  (`https://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd`) returns HTTP 400 with body
  `LOGOUT` — KRX has closed anonymous access to data.krx.co.kr (confirmed against both this page
  and a previously-public daily-price page as a control; corroborated by third-party reports of
  the same change). Not usable without a KRX member login, which this project doesn't have.
- A *different*, real, currently-working KIS TR was found instead: `FHPST01390000`
  ("변동성완화장치(VI) 현황", `/uapi/domestic-stock/v1/quotations/inquire-vi-status`), documented
  and maintained in KIS's own official examples repo (github.com/koreainvestment/open-trading-api,
  examples_llm/domestic_stock/inquire_vi_status/). Verified live 2026-09-09 against real dates
  (2026-05-27, 2026-09-09) and a per-ticker query (007120, 2026-05-27) that round-tripped the
  ticker's own `vi_count` sequence (1,2,3,4) correctly.

Response field mapping (from a live sample, `FHPST01390000` output rows):
    hts_kor_isnm    - 종목명 (Korean stock name)
    mksc_shrn_iscd  - 6-digit ticker
    bsop_date       - business date, YYYYMMDD
    cntg_vi_hour    - VI trigger time, HHMMSS (local KST wall-clock; can be in the 16:00-18:00
                       시간외 단일가매매 extended session, not just the 09:00-15:30 regular session)
    vi_cncl_hour    - VI release time, HHMMSS; empty string if not yet released
    vi_kind_code    - '1', '2', or '3'. Mapped from the *same TR family's* own request-parameter
                       docstring (`FID_RANK_SORT_CLS_CODE`: "0:전체 1:정적 2:동적 3:정적&동적"),
                       so '1' = static (정적) VI, '2' = dynamic (동적) VI, '3' = both together.
                       The '3' mapping was independently confirmed live (not just inferred from
                       the docstring): a sample row with vi_kind_code='3' (메지온/140410,
                       2019-06-28 15:30:22) had BOTH vi_stnd_prc (static reference price, 90700)
                       AND vi_dmc_stnd_prc (dynamic reference price, 82700) populated non-zero —
                       whereas a vi_kind_code='1' row on the same ticker/day had vi_stnd_prc
                       populated but vi_dmc_stnd_prc=0. This confirms '3' means both the static
                       and dynamic mechanisms triggered together, not an undocumented state.
    vi_count        - cumulative VI trigger count for that ticker on that business date (not
                       stored; upsert_vi_events's PK is (ticker, triggered_at) so this is
                       reconstructable from row count per ticker/day if ever needed)

Important scope caveat: this TR appears to be a "recent status board" style endpoint. A
market-wide, ticker-unscoped query (`FID_INPUT_ISCD=""`) returns at most 30 rows per call
(observed cap on 2026-05-27, a real high-volatility day). A per-ticker query
(`FID_INPUT_ISCD=<ticker>`) is NOT capped at 30 and returns that ticker's own full event list
for the date (verified: 007120 on 2026-05-27 returned exactly its own 4 events). The runner
(`run_vi_events_backfill.py`) therefore sweeps market-wide per day for efficiency, but flags any
(market, date) where the raw response hit exactly 30 rows and re-verifies those specific days with
per-ticker calls for our 200-ticker universe, to avoid silently dropping universe events on
capped days.
"""

VI_KIND_MAP = {"1": "STATIC", "2": "DYNAMIC", "3": "BOTH"}


def parse_vi_status_response(raw: dict, universe_tickers: set[str]) -> list[dict]:
    """Parse one FHPST01390000 response into vi_events rows, filtered to `universe_tickers`.

    Returns a list of dicts: {ticker, triggered_at, released_at, vi_type}, matching the
    `vi_events` table schema (ticker varchar(6), triggered_at/released_at timestamp, vi_type
    varchar(10)). `released_at` is None if the row's `vi_cncl_hour` is empty (VI not yet
    released — should not occur for historical dates, but handled defensively).
    """
    results = []
    for r in raw.get("output", []):
        ticker = r.get("mksc_shrn_iscd", "")
        if ticker not in universe_tickers:
            continue
        bsop_date = r["bsop_date"]  # YYYYMMDD
        date_str = f"{bsop_date[:4]}-{bsop_date[4:6]}-{bsop_date[6:8]}"

        trig_hms = r["cntg_vi_hour"]
        triggered_at = f"{date_str} {trig_hms[:2]}:{trig_hms[2:4]}:{trig_hms[4:6]}"

        rel_hms = r.get("vi_cncl_hour") or ""
        released_at = None
        # "000000" observed live 2026-09-09 (014680, 2026-06-08 09:02:29 trigger) on one row out
        # of 4926 -- a re-query moments later returned the correct non-zero release time (090342),
        # so this is a transient KIS response quirk, not a real "released at midnight" (VI cannot
        # occur outside 09:00-18:00 trading/extended hours). Treated the same as an empty string.
        if rel_hms and rel_hms != "000000":
            released_at = f"{date_str} {rel_hms[:2]}:{rel_hms[2:4]}:{rel_hms[4:6]}"

        kind_code = r.get("vi_kind_code", "")
        vi_type = VI_KIND_MAP.get(kind_code, f"UNKNOWN_{kind_code}")

        results.append({
            "ticker": ticker,
            "triggered_at": triggered_at,
            "released_at": released_at,
            "vi_type": vi_type,
        })
    return results


def raw_row_count(raw: dict) -> int:
    """Unfiltered row count in a response — used by the runner to detect the ~30-row cap."""
    return len(raw.get("output", []))
