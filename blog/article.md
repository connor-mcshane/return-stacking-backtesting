# Visualising Return Stacking in Streamlit

I'm a fan of Pensioncraft and was interested in Ramin's [return stacking video](https://www.youtube.com/watch?v=DCyOEaeM9wk). No free lunch, obviously, but it seemed like it could work as a core-light allocation alongside a regular equities portfolio. I wanted to pull the data myself and see what the numbers actually look like in Streamlit.

The first version pulled prices from Google Sheets. The charts looked fine. The numbers were wrong, though — I only noticed when I compared against Portfolio Visualizer.

---

## The Numbers Didn't Match

Same tickers, same weights, same date range. Should've been close.

| Portfolio | My App | Portfolio Visualizer |
|---|---|---|
| Capital Efficient (NTSX/GDE/GLD) | 16.1% CAGR | 21.4% CAGR |
| S&P 500 (SPY) | 13.9% CAGR | 16.4% CAGR |
| Traditional 60/40 (VTI/BND) | 13.0% CAGR | 15.8% CAGR |

Five percentage points off on CAGR. That's not a rounding error. That's a different story about whether a strategy works.

---

## Adjusted vs. Unadjusted Prices

Google Sheets' `GOOGLEFINANCE()` function returns raw close prices. Not adjusted close. When a stock pays a dividend, the raw close drops by the dividend amount the next day. If you compute returns from raw close, you're measuring price movement only — you're completely ignoring the cash that landed in the investor's account.

For something like GLD (gold, no dividends), this barely matters. But for SPY with a ~1.3% annual yield? That's real money. And for bond ETFs like BND or AGG with 3-4% yields? You're missing a huge chunk of the total return.

I'd flagged this as a known limitation early on but didn't prioritize it until the Portfolio Visualizer comparison made the gap obvious.

Fix: switch from Google Sheets to Yahoo Finance, which provides adjusted close prices that account for dividends and splits.

---

## yfinance SSL and Rate Limiting

`yfinance` is the standard Python library for Yahoo Finance data. Straightforward to use, unless you're on macOS with LibreSSL 2.8.

```
Failed to perform, curl: (60) SSL certificate problem:
self signed certificate in certificate chain
```

Newer versions of yfinance swapped their HTTP backend from `requests` to `curl_cffi`. That library ships its own SSL stack, which doesn't trust the certificate chain on older macOS setups. Setting `verify=False` on the session gets past it — but then Yahoo's rate limiter gets involved.

See, those SSL errors don't fail silently. Each one counts as a request. Twenty-six tickers, three retry attempts each, all failing on SSL before the data even loads? That's 78 requests that Yahoo remembers. And Yahoo's rate limit is IP-level and unforgiving.

```
YFRateLimitError: Too Many Requests. Rate limited. Try after a while.
```

"A while" turned out to be somewhere north of 30 minutes. I tried every workaround — batching, exponential backoff, different API endpoints, different yfinance versions. The older version that still uses `requests` worked around SSL, but by then the rate limit had already locked me out.

---

## Shelling Out to curl

While Python's `requests` library was getting 429s from Yahoo, plain `curl` from the terminal returned 200 every time.

```bash
curl -s "https://query2.finance.yahoo.com/v8/finance/chart/SPY?..." → 200 OK
```

Same URL. Same IP. Different HTTP client fingerprint. Yahoo was specifically fingerprinting and rate-limiting the Python HTTP client, not the IP itself.

So I rewrote the data fetcher to shell out to `curl` via `subprocess.run()`:

```python
result = subprocess.run(
    ["curl", "-s", "-H", "User-Agent: Mozilla/5.0", url],
    capture_output=True, text=True, timeout=30,
)
data = json.loads(result.stdout)
```

Not something you'd do in a real system, but for a side project that needs 26 tickers once a day it works fine and hasn't been rate-limited once.

---

## Validating the Fix

With Yahoo's adjusted close prices loaded, I re-ran the Portfolio Visualizer comparison:

| Portfolio | My App | Portfolio Visualizer | Gap |
|---|---|---|---|
| Capital Efficient | 21.4% CAGR | 21.4% CAGR | — |
| VTI (benchmark) | 15.9% CAGR | 15.8% CAGR | 0.1pp |
| S&P 500 (SPY) | $14,059 | $14,059 | $0 |

SPY's end balance matched to the dollar. VTI was off by a single basis point. The dividend adjustment was the entire gap.

One thing that still looks different: volatility and max drawdown. My app reports SPY volatility at 16.1% and max drawdown at -19.0%. Portfolio Visualizer shows 11.0% and -7.6%. That's not an error — PV computes from monthly returns, I compute from daily. Daily captures more intra-month fluctuation, so the numbers are naturally higher. Both are correct; they're just answering slightly different questions.

---

## Pre-Inception Phantom Data

Early on, the Return Stacked Core portfolio had a spike in its chart around January 2024. RSSB (Return Stacked Global Stocks & Bonds) had junk price data from Google Finance — $4 prices in December 2023 jumping to $21 in January 2024.

RSSB actually launched on January 2, 2024 at ~$21. Those $4 prices were phantom data that Google Finance returned for the pre-inception period. Computing returns across a jump from $4.40 to $21 gives you a +377% single-day return, which ripples through every portfolio metric.

The fix was a two-layer defense:

1. **Hardcoded inception dates** — filter out all data before a fund's known launch date
2. **Sanity filter** — drop any remaining rows with daily returns exceeding 100%

It sounds crude, and it is. But the alternative was trusting that every data source correctly handles pre-inception periods for newer ETFs, and they don't. Google Finance doesn't. Even Yahoo Finance can have quirks around IPO dates. A simple filter catches problems that no amount of clever API work will prevent.

---

The dashboard now lets you pick preset portfolios or build custom ones, compare against benchmarks, and toggle rebalancing frequencies. Data refreshes daily via a GitHub Action that fetches, validates, and uploads to GCS. The Streamlit app reads the parquet file over HTTPS.

In hindsight, I would have validated against Portfolio Visualizer on day one instead of building features on top of bad data for weeks. The dividend issue was the whole gap and it would have taken 30 minutes to catch.

`[IMAGE: screenshot of the final dashboard]`

*Code is on [GitHub](https://github.com/your-username/returnstacking-backtests). Live demo is on [Streamlit Cloud](https://your-app.streamlit.app).*
