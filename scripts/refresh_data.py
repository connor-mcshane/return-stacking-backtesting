"""
Refresh ETF price data from Yahoo Finance.

Downloads dividend-adjusted daily prices for all tracked tickers via the
Yahoo Finance chart API, validates the data, and saves a local parquet file.

Usage:
    python scripts/refresh_data.py

Requires:
    pip install pandas pyarrow
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "etf_prices.parquet"

ETF_TICKERS = [
    "VTI", "VXUS", "BND", "VNQ", "GLD",
    "RSST", "RSSB", "NTSX", "GDE",
    "DBMF", "KMLM", "CTA",
    "CAOS", "TAIL",
    "SPY", "TLT", "AGG", "IWM",
    "QQQ", "EFA", "VWO", "SCHH", "IAU", "GSG", "DBC",
    "VT",
]

START_DATE = "2008-01-01"
FETCH_DELAY = 0.4


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def _fetch_ticker(ticker: str) -> pd.DataFrame | None:
    period1 = int(datetime.strptime(START_DATE, "%Y-%m-%d").timestamp())
    period2 = int(time.time())

    url = (
        f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={period1}&period2={period2}&interval=1d&includeAdjustedClose=true"
    )

    for attempt in range(3):
        result = subprocess.run(
            ["curl", "-s", "-H", "User-Agent: Mozilla/5.0", url],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"  ✗ {ticker}: curl error")
            return None

        if "Too Many Requests" in result.stdout:
            wait = 2 ** (attempt + 2)
            print(f"    Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            print(f"  ✗ {ticker}: invalid JSON response")
            return None

        break
    else:
        print(f"  ✗ {ticker}: rate limited after retries")
        return None

    chart = data.get("chart", {}).get("result")
    if not chart:
        error = data.get("chart", {}).get("error", {})
        print(f"  ✗ {ticker}: {error.get('description', 'no data')}")
        return None

    chart = chart[0]
    timestamps = chart.get("timestamp", [])
    if not timestamps:
        print(f"  ✗ {ticker}: no timestamps")
        return None

    quote = chart["indicators"]["quote"][0]
    adjclose_list = chart["indicators"].get("adjclose", [{}])[0].get("adjclose", [])

    df = pd.DataFrame({
        "ticker": ticker,
        "date": pd.to_datetime([datetime.utcfromtimestamp(ts) for ts in timestamps]).normalize(),
        "open": quote.get("open"),
        "high": quote.get("high"),
        "low": quote.get("low"),
        "close": quote.get("close"),
        "adj_close": adjclose_list if adjclose_list else quote.get("close"),
        "volume": quote.get("volume"),
    })

    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "adj_close"])
    return df


def fetch_all() -> pd.DataFrame:
    print(f"Downloading {len(ETF_TICKERS)} tickers from Yahoo Finance...")

    frames = []
    for i, ticker in enumerate(ETF_TICKERS):
        print(f"  [{i + 1}/{len(ETF_TICKERS)}] {ticker}", end=" ")
        df = _fetch_ticker(ticker)
        if df is not None and not df.empty:
            frames.append(df)
            print(f"→ {len(df):,} rows")
        elif df is None:
            pass
        else:
            print("→ 0 rows")

        if i < len(ETF_TICKERS) - 1:
            time.sleep(FETCH_DELAY)

    if not frames:
        raise SystemExit("No data fetched from any ticker.")

    combined = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    print(f"\nTotal: {len(combined):,} rows across {combined['ticker'].nunique()} tickers")
    return combined


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame) -> bool:
    """Check data quality. Returns True if all checks pass."""
    ok = True
    today = pd.Timestamp.now().normalize()
    # Allow up to 5 calendar days of staleness (weekends + holidays)
    stale_cutoff = today - timedelta(days=5)

    fetched_tickers = set(df["ticker"].unique())
    expected_tickers = set(ETF_TICKERS)

    missing = expected_tickers - fetched_tickers
    if missing:
        print(f"\n  FAIL: Missing tickers: {sorted(missing)}")
        ok = False

    nan_counts = df.groupby("ticker")["adj_close"].apply(lambda s: s.isna().sum())
    bad_nan = nan_counts[nan_counts > 0]
    if not bad_nan.empty:
        print(f"\n  FAIL: Tickers with NaN adj_close: {dict(bad_nan)}")
        ok = False

    print("\n  Ticker freshness:")
    for ticker in sorted(fetched_tickers):
        t_data = df[df["ticker"] == ticker]
        last_date = t_data["date"].max()
        rows = len(t_data)
        fresh = "OK" if last_date >= stale_cutoff else "STALE"
        if fresh == "STALE":
            ok = False
        print(f"    {ticker:6s}  {rows:>5,} rows  last={last_date.date()}  [{fresh}]")

    if ok:
        print("\n  All validation checks passed.")
    else:
        print("\n  VALIDATION FAILED — see errors above.")

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = fetch_all()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    size_mb = OUTPUT_PATH.stat().st_size / 1_048_576
    print(f"\nWrote {OUTPUT_PATH} ({size_mb:.1f} MB)")

    passed = validate(df)

    if not passed:
        print("\nValidation failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
