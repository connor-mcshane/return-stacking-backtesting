"""Return Stacking Portfolio Monitor — Streamlit Dashboard."""

from __future__ import annotations

from pathlib import Path

import empyrical as ep
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Return Stacking Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "data" / "etf_prices.parquet"

# (name, asset_class, inception_date)
TICKER_INFO = {
    "VTI":  ("Vanguard Total US Stock Market",                "US Equities",    "2001-05-24"),
    "SPY":  ("SPDR S&P 500",                                  "US Equities",    "1993-01-22"),
    "QQQ":  ("Invesco Nasdaq-100",                             "US Equities",    "1999-03-10"),
    "IWM":  ("iShares Russell 2000 Small-Cap",                 "US Equities",    "2000-05-22"),
    "VXUS": ("Vanguard Total International Stock",             "International",  "2011-01-26"),
    "EFA":  ("iShares MSCI EAFE (Developed ex-US)",            "International",  "2001-08-14"),
    "VWO":  ("Vanguard FTSE Emerging Markets",                 "International",  "2005-03-04"),
    "BND":  ("Vanguard Total US Bond Market",                  "Bonds",          "2007-04-03"),
    "AGG":  ("iShares Core US Aggregate Bond",                 "Bonds",          "2003-09-22"),
    "TLT":  ("iShares 20+ Year Treasury Bond",                 "Bonds",          "2002-07-22"),
    "VNQ":  ("Vanguard Real Estate (REITs)",                   "Real Estate",    "2004-09-23"),
    "SCHH": ("Schwab US REIT",                                 "Real Estate",    "2011-01-13"),
    "GLD":  ("SPDR Gold Shares",                               "Commodities",    "2004-11-18"),
    "IAU":  ("iShares Gold Trust",                             "Commodities",    "2005-01-21"),
    "GSG":  ("iShares S&P GSCI Commodity Index",               "Commodities",    "2006-07-10"),
    "DBC":  ("Invesco DB Commodity Tracking",                  "Commodities",    "2006-02-03"),
    "RSST": ("Return Stacked US Stocks & Managed Futures",     "Return Stacked", "2023-09-05"),
    "RSSB": ("Return Stacked Global Stocks & Bonds",           "Return Stacked", "2024-01-02"),
    "NTSX": ("WisdomTree 90/60 US Equity/Treasury",            "Capital Efficient", "2018-08-02"),
    "GDE":  ("WisdomTree Efficient Gold + Equity",             "Capital Efficient", "2022-03-17"),
    "DBMF": ("iMGP DBi Managed Futures",                      "Managed Futures", "2019-05-07"),
    "KMLM": ("KFA Mount Lucas Managed Futures",                "Managed Futures", "2020-12-01"),
    "CTA":  ("Simplify Managed Futures",                       "Managed Futures", "2022-03-07"),
    "CAOS": ("Alpha Architect Tail Risk (Put Options)",        "Tail Risk",      "2019-12-17"),
    "TAIL": ("Cambria Tail Risk ETF (Put Options)",            "Tail Risk",      "2017-04-06"),
    "VT":   ("Vanguard Total World Stock",                     "Global Equities","2008-06-24"),
}

PORTFOLIOS = {
    "Traditional 60/40": {"VTI": 0.60, "BND": 0.40},
    "Global 60/40": {"VTI": 0.30, "VXUS": 0.30, "BND": 0.40},
    "Return Stacked Core": {"RSST": 0.50, "RSSB": 0.50},
    "Capital Efficient": {"NTSX": 0.80, "GDE": 0.10, "GLD": 0.10},
    "All Weather (DIY)": {"VTI": 0.30, "TLT": 0.40, "GLD": 0.15, "DBC": 0.15},
    "Max Diversification": {
        "VTI": 0.25, "VXUS": 0.15, "DBMF": 0.20,
        "BND": 0.15, "VNQ": 0.10, "GLD": 0.10, "TAIL": 0.05,
    },
}

BENCHMARKS = {
    "S&P 500 (SPY)": "SPY",
    "Global Equity (VT)": "VT",
}

ASSET_CLASSES = {
    "US Equities": ["VTI", "SPY", "QQQ", "IWM"],
    "Global Equities": ["VT"],
    "International": ["VXUS", "EFA", "VWO"],
    "Bonds": ["BND", "AGG", "TLT"],
    "Real Estate": ["VNQ", "SCHH"],
    "Commodities": ["GLD", "IAU", "GSG", "DBC"],
    "Return Stacked": ["RSST", "RSSB"],
    "Capital Efficient": ["NTSX", "GDE"],
    "Managed Futures": ["DBMF", "KMLM", "CTA"],
    "Tail Risk": ["CAOS", "TAIL"],
}

REBALANCE_OPTIONS = {
    "Never": None,
    "Annually": 252,
    "Quarterly": 63,
    "Monthly": 21,
}


def ticker_label(ticker: str) -> str:
    info = TICKER_INFO.get(ticker)
    return f"{ticker} — {info[0]}" if info else ticker


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading price data...")
def load_prices() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(
            "No data file found. "
            "Run `python scripts/refresh_data.py` to generate data/etf_prices.parquet."
        )
        st.stop()
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"])


@st.cache_data
def load_returns() -> pd.DataFrame:
    df = load_prices()
    df = df.sort_values(["ticker", "date"])

    # Filter out pre-inception data using known fund launch dates.
    # GOOGLEFINANCE can return junk prices from before a fund was listed.
    inception_dates = {
        t: pd.Timestamp(info[2]) for t, info in TICKER_INFO.items() if len(info) >= 3
    }
    mask = pd.Series(True, index=df.index)
    for ticker, inception in inception_dates.items():
        mask &= ~((df["ticker"] == ticker) & (df["date"] < inception))
    df = df[mask]

    df["simple_return"] = df.groupby("ticker")["adj_close"].pct_change()
    df = df.dropna(subset=["simple_return"])

    # Safety net: also drop any remaining rows with unrealistic returns (>100%/day)
    # in case of data issues not covered by inception dates.
    df = df[df["simple_return"].abs() <= 1.0]

    return df


@st.cache_data
def get_ticker_date_ranges() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """First and last date with return data for each ticker."""
    df = load_returns()
    ranges = {}
    for ticker, grp in df.groupby("ticker"):
        ranges[ticker] = (grp["date"].min(), grp["date"].max())
    return ranges


# ---------------------------------------------------------------------------
# Date alignment
# ---------------------------------------------------------------------------

def align_portfolio_returns(
    returns_wide: pd.DataFrame, weights: dict
) -> tuple[pd.DataFrame, list[str], pd.Timestamp | None]:
    """
    Trim returns_wide to the common date range where ALL portfolio
    constituents have data.  Returns (trimmed_df, available_tickers, common_start).
    """
    tickers = [t for t in weights if t in returns_wide.columns]
    if not tickers:
        return returns_wide, [], None

    subset = returns_wide[tickers].dropna(how="any")
    if subset.empty:
        return subset, tickers, None

    return subset, tickers, subset.index.min()


# ---------------------------------------------------------------------------
# Portfolio math (using empyrical)
# ---------------------------------------------------------------------------

def compute_portfolio_returns(returns_aligned: pd.DataFrame, weights: dict) -> pd.Series:
    """Weighted portfolio daily returns from already-aligned data."""
    tickers = [t for t in weights if t in returns_aligned.columns]
    if not tickers:
        return pd.Series(dtype=float)
    w = pd.Series({t: weights[t] for t in tickers})
    w = w / w.sum()
    return returns_aligned[tickers].mul(w).sum(axis=1)


def simulate_portfolio_value(
    returns_aligned: pd.DataFrame,
    weights: dict,
    initial_balance: float,
    rebalance_days: int | None = None,
) -> pd.DataFrame:
    """
    Simulate portfolio value in dollar terms with optional periodic rebalancing.
    Expects already-aligned (no NaN) return data.
    """
    tickers = [t for t in weights if t in returns_aligned.columns]
    if not tickers:
        return pd.DataFrame()
    target_w = pd.Series({t: weights[t] for t in tickers})
    target_w = target_w / target_w.sum()

    rets = returns_aligned[tickers]
    dates = rets.index
    n_days = len(dates)

    holdings = target_w * initial_balance
    values = []

    for i in range(n_days):
        day_rets = rets.iloc[i]
        holdings = holdings * (1 + day_rets)
        portfolio_val = holdings.sum()
        values.append({"date": dates[i], "portfolio_value": portfolio_val})

        if rebalance_days and (i + 1) % rebalance_days == 0 and i < n_days - 1:
            holdings = target_w * portfolio_val

    return pd.DataFrame(values).set_index("date")


def compute_metrics(daily_returns: pd.Series) -> dict:
    if len(daily_returns) < 2:
        return {}
    return {
        "CAGR": ep.annual_return(daily_returns, period="daily"),
        "Total Return": ep.cum_returns_final(daily_returns),
        "Ann. Volatility": ep.annual_volatility(daily_returns, period="daily"),
        "Sharpe": ep.sharpe_ratio(daily_returns, period="daily"),
        "Sortino": ep.sortino_ratio(daily_returns, period="daily"),
        "Max Drawdown": ep.max_drawdown(daily_returns),
        "Calmar": ep.calmar_ratio(daily_returns, period="daily"),
    }


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

returns_df = load_returns()

if returns_df.empty:
    st.error("No data found. Run `python scripts/refresh_data.py` to generate the data file.")
    st.stop()

returns_wide = returns_df.pivot(index="date", columns="ticker", values="simple_return")
available_tickers = sorted(returns_wide.columns.tolist())
ticker_ranges = get_ticker_date_ranges()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Portfolio Monitor")
st.sidebar.markdown("---")

selected_portfolios = st.sidebar.multiselect(
    "Compare portfolios",
    list(PORTFOLIOS.keys()),
    default=["Traditional 60/40", "Return Stacked Core", "All Weather (DIY)"],
)

selected_benchmarks = st.sidebar.multiselect(
    "Benchmarks",
    list(BENCHMARKS.keys()),
    default=["S&P 500 (SPY)"],
)

# --- Custom portfolio builder ---
with st.sidebar.expander("Build a custom portfolio"):
    custom_tickers = st.multiselect(
        "Select tickers",
        available_tickers,
        default=[],
        format_func=ticker_label,
        key="custom_tickers",
    )
    custom_weights: dict[str, float] = {}
    if custom_tickers:
        even_weight = round(100.0 / len(custom_tickers), 1)
        for t in custom_tickers:
            custom_weights[t] = st.slider(
                ticker_label(t), 0.0, 100.0, even_weight, 0.5, key=f"cw_{t}",
            ) / 100.0
        total_w = sum(custom_weights.values())
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"Weights sum to {total_w:.1%} — they'll be normalized to 100 %.")

has_custom = bool(custom_weights)

st.sidebar.markdown("---")
initial_balance = st.sidebar.number_input(
    "Initial investment ($)",
    min_value=100,
    max_value=10_000_000,
    value=10_000,
    step=1_000,
    format="%d",
)

rebalance_freq = st.sidebar.selectbox(
    "Rebalancing frequency",
    list(REBALANCE_OPTIONS.keys()),
    index=0,
)
rebalance_days = REBALANCE_OPTIONS[rebalance_freq]

LOOKBACK_OPTIONS = {
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "YTD": None,
    "1Y": 252,
    "3Y": 756,
    "5Y": 1260,
    "10Y": 2520,
    "Max": None,
}

st.sidebar.markdown("---")
lookback = st.sidebar.radio(
    "Lookback window",
    list(LOOKBACK_OPTIONS.keys()),
    index=4,
    horizontal=True,
)

max_date = returns_wide.index.max()
if lookback == "YTD":
    start_date = pd.Timestamp(f"{max_date.year}-01-01")
elif lookback == "Max":
    start_date = returns_wide.index.min()
else:
    trading_days = LOOKBACK_OPTIONS[lookback]
    start_date = max_date - pd.tseries.offsets.BDay(trading_days)

returns_filtered = returns_wide.loc[start_date:]

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Showing: {returns_filtered.index.min().strftime('%b %d, %Y')} — "
    f"{returns_filtered.index.max().strftime('%b %d, %Y')} "
    f"({len(returns_filtered)} trading days)"
)
st.sidebar.caption(f"{len(available_tickers)} tickers | {len(returns_df):,} total observations")

# ---------------------------------------------------------------------------
# Build unified entry list: preset portfolios + custom portfolio + benchmarks
# ---------------------------------------------------------------------------

ALL_ENTRIES: dict[str, dict[str, float]] = {}
ENTRY_STYLE: dict[str, str] = {}  # "solid" or "dash"

for name in selected_portfolios:
    ALL_ENTRIES[name] = PORTFOLIOS[name]
    ENTRY_STYLE[name] = "solid"

if has_custom:
    ALL_ENTRIES["Custom Portfolio"] = custom_weights
    ENTRY_STYLE["Custom Portfolio"] = "solid"

for bm_label in selected_benchmarks:
    ticker = BENCHMARKS[bm_label]
    ALL_ENTRIES[bm_label] = {ticker: 1.0}
    ENTRY_STYLE[bm_label] = "dash"

# ---------------------------------------------------------------------------
# Compute data for every entry (with date alignment)
# ---------------------------------------------------------------------------

portfolio_aligned = {}
portfolio_daily = {}
portfolio_cum = {}
portfolio_values = {}
portfolio_warnings = {}

for name, weights in ALL_ENTRIES.items():
    aligned, used_tickers, common_start = align_portfolio_returns(returns_filtered, weights)

    missing_tickers = [t for t in weights if t not in returns_wide.columns]
    limited_tickers = []
    for t in used_tickers:
        if t in ticker_ranges:
            t_start = ticker_ranges[t][0]
            if t_start > returns_filtered.index.min():
                limited_tickers.append(
                    f"{t} (from {t_start.strftime('%b %Y')})"
                )

    warnings = []
    if missing_tickers:
        warnings.append(f"No data for: {', '.join(missing_tickers)}")
    if limited_tickers:
        warnings.append(f"Limited history: {', '.join(limited_tickers)}")
    if common_start and common_start > returns_filtered.index.min():
        warnings.append(
            f"Comparison starts {common_start.strftime('%b %d, %Y')} "
            f"(earliest date with data for all holdings)"
        )
    portfolio_warnings[name] = warnings

    portfolio_aligned[name] = aligned
    daily_rets = compute_portfolio_returns(aligned, weights)
    portfolio_daily[name] = daily_rets
    portfolio_cum[name] = ep.cum_returns(daily_rets) if not daily_rets.empty else pd.Series(dtype=float)
    portfolio_values[name] = simulate_portfolio_value(
        aligned, weights, initial_balance, rebalance_days,
    )

all_entry_names = list(ALL_ENTRIES.keys())


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Return Stacking Portfolio Monitor")

# ---- Data availability warnings ----
for name in all_entry_names:
    warnings = portfolio_warnings.get(name, [])
    if warnings:
        with st.expander(f"⚠ {name} — data coverage note", expanded=False):
            for w in warnings:
                st.caption(w)

# ---- KPI cards ----
if all_entry_names:
    st.subheader("Portfolio KPIs")
    cols = st.columns(min(len(all_entry_names), 5))
    for i, name in enumerate(all_entry_names):
        m = compute_metrics(portfolio_daily[name])
        val_df = portfolio_values[name]
        final_val = val_df["portfolio_value"].iloc[-1] if not val_df.empty else initial_balance
        with cols[i % len(cols)]:
            st.markdown(f"**{name}**")
            st.metric("Final Value", f"${final_val:,.0f}", f"{m.get('Total Return', 0):+.1%}")
            c1, c2 = st.columns(2)
            c1.metric("CAGR", f"{m.get('CAGR', 0):.1%}")
            c2.metric("Sharpe", f"{m.get('Sharpe', 0):.2f}")
            c1.metric("Max DD", f"{m.get('Max Drawdown', 0):.1%}")
            c2.metric("Volatility", f"{m.get('Ann. Volatility', 0):.1%}")

st.markdown("---")

# ---- Tab layout ----
tab_value, tab_perf, tab_dd, tab_corr, tab_assets, tab_detail, tab_ref = st.tabs([
    "Portfolio Value", "Cumulative Returns", "Drawdowns",
    "Correlations", "Asset Classes", "Detailed Metrics", "ETF Reference",
])

# ---- Portfolio Value tab (dollar terms) ----
with tab_value:
    if portfolio_values:
        fig = go.Figure()
        for name, val_df in portfolio_values.items():
            if val_df.empty:
                continue
            fig.add_trace(go.Scatter(
                x=val_df.index, y=val_df["portfolio_value"],
                name=name, mode="lines",
                line=dict(dash=ENTRY_STYLE.get(name, "solid")),
            ))
        fig.add_hline(
            y=initial_balance, line_dash="dash", line_color="gray",
            annotation_text=f"Initial ${initial_balance:,.0f}",
        )
        rebal_label = f" | Rebalanced {rebalance_freq.lower()}" if rebalance_days else " | No rebalancing"
        fig.update_layout(
            title=f"Portfolio Value (${initial_balance:,.0f} invested){rebal_label}",
            yaxis_title="Value ($)",
            yaxis_tickprefix="$",
            yaxis_tickformat=",",
            xaxis_title="",
            hovermode="x unified",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Final Values")
        summary_rows = []
        for name in all_entry_names:
            val_df = portfolio_values[name]
            if val_df.empty:
                continue
            final = val_df["portfolio_value"].iloc[-1]
            gain = final - initial_balance
            m = compute_metrics(portfolio_daily[name])
            aligned_start = portfolio_aligned[name].index.min()
            entry_type = "Benchmark" if ENTRY_STYLE.get(name) == "dash" else "Portfolio"
            summary_rows.append({
                "Type": entry_type,
                "Portfolio": name,
                "Data From": aligned_start.strftime("%b %d, %Y"),
                "Final Value": f"${final:,.0f}",
                "Gain/Loss": f"${gain:+,.0f}",
                "Total Return": f"{m.get('Total Return', 0):.1%}",
                "CAGR": f"{m.get('CAGR', 0):.1%}",
                "Rebalancing": rebalance_freq,
            })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    allocation_entries = [n for n in all_entry_names if len(ALL_ENTRIES[n]) > 1]
    if allocation_entries:
        st.subheader("Target Allocations")
        pie_cols = st.columns(min(len(allocation_entries), 3))
        for i, name in enumerate(allocation_entries):
            with pie_cols[i % 3]:
                w = ALL_ENTRIES[name]
                labels = [ticker_label(t) for t in w.keys()]
                fig = px.pie(
                    names=labels, values=list(w.values()),
                    title=name, hole=0.4,
                )
                fig.update_layout(height=350, margin=dict(t=40, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)

# ---- Cumulative Returns tab (percentage) ----
with tab_perf:
    if portfolio_cum:
        fig = go.Figure()
        for name, cum in portfolio_cum.items():
            if cum.empty:
                continue
            fig.add_trace(go.Scatter(
                x=cum.index, y=cum.values * 100,
                name=name, mode="lines",
                line=dict(dash=ENTRY_STYLE.get(name, "solid")),
            ))
        fig.update_layout(
            title="Cumulative Returns",
            yaxis_title="Return (%)",
            xaxis_title="",
            hovermode="x unified",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

# ---- Drawdowns tab ----
with tab_dd:
    if portfolio_daily:
        fig = go.Figure()
        for name, rets in portfolio_daily.items():
            if rets.empty:
                continue
            dd = ep.cum_returns(rets)
            wealth = 1 + dd
            peak = wealth.cummax()
            drawdown = ((wealth - peak) / peak) * 100
            is_bench = ENTRY_STYLE.get(name) == "dash"
            fig.add_trace(go.Scatter(
                x=drawdown.index, y=drawdown.values,
                name=name, mode="lines",
                line=dict(dash="dash" if is_bench else "solid"),
                fill=None if is_bench else "tozeroy",
            ))
        fig.update_layout(
            title="Portfolio Drawdowns",
            yaxis_title="Drawdown (%)",
            hovermode="x unified",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

# ---- Correlations tab ----
with tab_corr:
    default_corr = [t for t in ["VTI", "TLT", "GLD", "DBMF", "RSST", "VNQ"] if t in available_tickers]
    corr_tickers = st.multiselect(
        "Select tickers for correlation matrix",
        available_tickers,
        default=default_corr,
        format_func=ticker_label,
    )

    corr_window = st.slider("Rolling window (days)", 30, 252, 60)

    if len(corr_tickers) >= 2:
        corr_data = returns_filtered[corr_tickers].dropna()

        if not corr_data.empty:
            st.caption(
                f"Correlation computed on {len(corr_data)} overlapping trading days "
                f"({corr_data.index.min().strftime('%b %Y')} — {corr_data.index.max().strftime('%b %Y')})"
            )

            corr_matrix = corr_data.corr()
            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title=f"Correlation Matrix ({lookback})",
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader(f"Rolling {corr_window}-Day Correlation vs VTI")
            if "VTI" in corr_tickers:
                others = [t for t in corr_tickers if t != "VTI"]
                fig2 = go.Figure()
                for t in others:
                    rolling_corr = corr_data["VTI"].rolling(corr_window).corr(corr_data[t])
                    fig2.add_trace(go.Scatter(
                        x=rolling_corr.index, y=rolling_corr.values,
                        name=ticker_label(t), mode="lines",
                    ))
                fig2.add_hline(y=0, line_dash="dash", line_color="gray")
                fig2.update_layout(
                    yaxis_title="Correlation",
                    hovermode="x unified",
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No overlapping data for the selected tickers in this time range.")

# ---- Asset classes tab ----
with tab_assets:
    st.subheader("Performance by Asset Class")

    asset_metrics = []
    for cls, tickers in ASSET_CLASSES.items():
        valid = [t for t in tickers if t in returns_filtered.columns]
        for t in valid:
            rets = returns_filtered[t].dropna()
            if len(rets) < 2:
                continue
            m = compute_metrics(rets)
            info = TICKER_INFO.get(t, (t, cls))
            t_start = ticker_ranges.get(t, (None, None))[0]
            data_from = t_start.strftime("%b %Y") if t_start else "—"
            m["Ticker"] = t
            m["Name"] = info[0]
            m["Asset Class"] = cls
            m["Data From"] = data_from
            m["Days"] = len(rets)
            asset_metrics.append(m)

    if asset_metrics:
        metrics_df = pd.DataFrame(asset_metrics)
        metrics_df = metrics_df[["Asset Class", "Ticker", "Name", "Data From", "Days",
                                  "CAGR", "Ann. Volatility", "Sharpe", "Max Drawdown", "Total Return"]]

        for col in ["CAGR", "Ann. Volatility", "Max Drawdown", "Total Return"]:
            metrics_df[col] = metrics_df[col].map(lambda x: f"{x:.1%}")
        metrics_df["Sharpe"] = metrics_df["Sharpe"].map(lambda x: f"{x:.2f}")

        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    scatter_data = []
    for cls, tickers in ASSET_CLASSES.items():
        valid = [t for t in tickers if t in returns_filtered.columns]
        for t in valid:
            rets = returns_filtered[t].dropna()
            if len(rets) < 2:
                continue
            m = compute_metrics(rets)
            scatter_data.append({
                "Ticker": t, "Asset Class": cls,
                "CAGR": m.get("CAGR", 0) * 100,
                "Volatility": m.get("Ann. Volatility", 0) * 100,
            })
    if scatter_data:
        sdf = pd.DataFrame(scatter_data)
        fig = px.scatter(
            sdf, x="Volatility", y="CAGR",
            color="Asset Class", text="Ticker",
            title="Risk vs Return (annualized)",
            labels={"Volatility": "Annualized Volatility (%)", "CAGR": "CAGR (%)"},
        )
        fig.update_traces(textposition="top center", marker=dict(size=10))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ---- Detailed metrics tab ----
with tab_detail:
    st.subheader("Side-by-Side Portfolio Comparison")

    if all_entry_names:
        comparison = {}
        for name in all_entry_names:
            comparison[name] = compute_metrics(portfolio_daily[name])

        comp_df = pd.DataFrame(comparison).T
        fmt_df = comp_df.copy()
        for col in ["CAGR", "Total Return", "Ann. Volatility", "Max Drawdown"]:
            if col in fmt_df.columns:
                fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.2%}")
        for col in ["Sharpe", "Sortino", "Calmar"]:
            if col in fmt_df.columns:
                fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.2f}")

        st.dataframe(fmt_df, use_container_width=True)

    st.subheader("Rolling 1-Year Returns")
    if portfolio_daily:
        fig = go.Figure()
        for name, rets in portfolio_daily.items():
            if rets.empty:
                continue
            rolling_1y = rets.rolling(252).apply(lambda x: (1 + x).prod() - 1, raw=False)
            fig.add_trace(go.Scatter(
                x=rolling_1y.index, y=rolling_1y.values * 100,
                name=name, mode="lines",
                line=dict(dash=ENTRY_STYLE.get(name, "solid")),
            ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            yaxis_title="Rolling 1Y Return (%)",
            hovermode="x unified",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

# ---- ETF Reference tab ----
with tab_ref:
    st.subheader("ETF Reference Guide")
    st.markdown("All ETFs used in this dashboard, grouped by asset class.")

    for cls in ASSET_CLASSES:
        tickers_in_class = ASSET_CLASSES[cls]
        rows = []
        for t in tickers_in_class:
            info = TICKER_INFO.get(t, (t, cls, ""))
            inception = info[2] if len(info) >= 3 else "—"
            t_range = ticker_ranges.get(t)
            if t_range:
                data_range = f"{t_range[0].strftime('%b %Y')} — {t_range[1].strftime('%b %Y')}"
            else:
                data_range = "No data"
            rows.append({
                "Ticker": t,
                "Name": info[0],
                "Inception": inception,
                "Data Range": data_range,
            })
        st.markdown(f"**{cls}**")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
