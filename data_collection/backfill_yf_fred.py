from datetime import datetime, timedelta
import pandas as pd
import psycopg2
import yfinance as yf
import pandas_datareader.data as web

YF_SYMBOLS = {"^GSPC": "snp500_close", "^IXIC": "nasdaq_close", "^SOX": "phlx_semi_close",
              "^VIX": "vix", "CL=F": "wti_crude_oil", "GC=F": "gold_price"}
FRED_SERIES = {"DEXKOUS": "usd_krw", "DGS10": "us_10y_yield",
               "DFEDTARL": "fed_rate", "INTDSRKRM193N": "kr_base_rate"}


def shift_us_date_to_kr(us_date: str) -> str:
    d = datetime.strptime(us_date, "%Y-%m-%d")
    return (d + timedelta(days=1)).strftime("%Y-%m-%d")


def fetch_yfinance(start: str, end: str) -> dict:
    out = {}
    for symbol, column in YF_SYMBOLS.items():
        df = yf.download(symbol, start=start, end=end, progress=False)["Close"]
        # yfinance returns a DataFrame with symbol as column when downloading multiple symbols
        # Extract the specific symbol column if it's a DataFrame, otherwise use the Series directly
        if isinstance(df, pd.DataFrame):
            series = df[symbol]
        else:
            series = df
        for us_date, value in series.items():
            # Handle both string and timestamp types
            if isinstance(us_date, str):
                us_date_str = us_date
            else:
                us_date_str = us_date.strftime("%Y-%m-%d")
            kr_date = shift_us_date_to_kr(us_date_str)
            out.setdefault(kr_date, {})[column] = float(value)
    return out


def fetch_fred(start: str, end: str) -> dict:
    out = {}
    for series, column in FRED_SERIES.items():
        df = web.DataReader(series, "fred", start, end)[series].dropna()
        for us_date, value in df.items():
            # Handle both string and timestamp types
            if isinstance(us_date, str):
                us_date_str = us_date
            else:
                us_date_str = us_date.strftime("%Y-%m-%d")
            kr_date = shift_us_date_to_kr(us_date_str)
            out.setdefault(kr_date, {})[column] = float(value)
    return out


def upsert_market_global(dsn: str, trade_date: str, row: dict) -> None:
    if not row:
        return
    columns = ", ".join(row.keys())
    placeholders = ", ".join(f"%({k})s" for k in row.keys())
    updates = ", ".join(f"{k} = COALESCE(EXCLUDED.{k}, market_global.{k})" for k in row.keys())
    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO market_global (trade_date, {columns})
                VALUES (%(trade_date)s, {placeholders})
                ON CONFLICT (trade_date) DO UPDATE SET {updates}""",
            {**row, "trade_date": trade_date},
        )
    conn.commit()
    conn.close()
