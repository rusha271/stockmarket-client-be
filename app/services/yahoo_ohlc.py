"""Fetch OHLC data from Yahoo Finance for NSE symbols."""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

# Number of 5-min candles to use as model input
CANDLES_NEEDED = 40
# Days of history to request (Yahoo 5m usually allows up to 7d)
FETCH_DAYS = 6

# NSE symbol -> Yahoo ticker (without .NS). Use when Yahoo uses a different symbol.
YAHOO_SYMBOL_OVERRIDES = {
    "LTIMINDST": "LTIM",  # LTIMindtree: Yahoo uses LTIM, NSE uses LTIM
}

# If primary NSE ticker fails on Yahoo, try these full Yahoo symbols (e.g. BSE listing).
YAHOO_FALLBACK_TICKERS: dict[str, tuple[str, ...]] = {
    "TATAMOTORS": ("TATAMOTORS.BO",),
}


def _normalize_nse(symbol: str) -> str:
    return (symbol or "").strip().upper()


def _ensure_nse_symbol(symbol: str) -> str:
    """Return Yahoo Finance ticker (e.g. SYMBOL.NS). Uses overrides when Yahoo symbol differs from NSE."""
    s = _normalize_nse(symbol)
    if not s:
        return s
    base = YAHOO_SYMBOL_OVERRIDES.get(s, s)
    return base if base.endswith(".NS") else f"{base}.NS"


def _suppress_yfinance_noise() -> None:
    for name in ("yfinance", "urllib3.connectionpool", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _slice_first_ticker_ohlc(data: Any) -> Any:
    """Single yf.download result → DataFrame with Open, High, Low, Close columns."""
    if pd is None or data is None or data.empty:
        return data
    if isinstance(data.columns, pd.MultiIndex):
        tickers = data.columns.get_level_values(0).unique()
        if len(tickers) >= 1:
            return data[tickers[0]]
    return data


def _rows_from_ohlc_df(df: Any, last_n: int) -> list[dict[str, float]]:
    """Build OHLC rows from a Yahoo OHLC DataFrame (flat columns)."""
    if pd is None or df is None or df.empty:
        return []
    o_col = "Open" if "Open" in df.columns else df.columns[0]
    h_col = "High" if "High" in df.columns else df.columns[min(1, len(df.columns) - 1)]
    l_col = "Low" if "Low" in df.columns else df.columns[min(2, len(df.columns) - 1)]
    c_col = "Close" if "Close" in df.columns else df.columns[min(3, len(df.columns) - 1)]

    rows: list[dict[str, float]] = []
    for _, row in df.iterrows():
        try:
            o, h, l, c = float(row[o_col]), float(row[h_col]), float(row[l_col]), float(row[c_col])
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(x) for x in (o, h, l, c)):
            continue
        rows.append({"open": o, "high": h, "low": l, "close": c})

    if len(rows) < 8:
        return []

    return rows[-last_n:] if len(rows) >= last_n else rows


def _download_ohlc_raw(ticker: str, days: int, interval: str) -> Any:
    import yfinance as yf

    _suppress_yfinance_noise()
    period = f"{days}d"
    return yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
        prepost=False,
        threads=False,
    )


def fetch_ohlc(
    symbol: str,
    days: int = FETCH_DAYS,
    interval: str = "5m",
    last_n: int = CANDLES_NEEDED,
) -> list[dict[str, float]]:
    """
    Fetch OHLC for symbol (NSE: use .NS), return last_n 5-min candles.
    Each candle: {"open": float, "high": float, "low": float, "close": float}.
    """
    try:
        import yfinance as yf  # noqa: F401 — ensures installed
    except ImportError:
        raise RuntimeError("yfinance is required; install with: pip install yfinance")

    sym = _normalize_nse(symbol)
    tickers_to_try: list[str] = [_ensure_nse_symbol(sym)]
    for alt in YAHOO_FALLBACK_TICKERS.get(sym, ()):
        if alt not in tickers_to_try:
            tickers_to_try.append(alt)

    last_err: Exception | None = None
    for ticker in tickers_to_try:
        if not ticker:
            continue
        try:
            data = _download_ohlc_raw(ticker, days, interval)
            df = _slice_first_ticker_ohlc(data)
            rows = _rows_from_ohlc_df(df, last_n)
            if len(rows) >= 8:
                return rows
        except Exception as e:
            last_err = e
            logger.debug("yfinance download failed for %s: %s", ticker, e)

    if last_err:
        logger.warning("yfinance download failed for %s after fallbacks: %s", symbol, last_err)
    raise ValueError(f"No OHLC data returned for {symbol}")


def _candles_for_yahoo_ticker(data: Any, yahoo_ticker: str, last_n: int) -> list[dict[str, float]]:
    """Extract candles for one Yahoo ticker from a multi-ticker download."""
    if pd is None or data is None or data.empty:
        return []
    if not isinstance(data.columns, pd.MultiIndex):
        return []
    if yahoo_ticker not in data.columns.get_level_values(0):
        return []
    sub = data[yahoo_ticker]
    return _rows_from_ohlc_df(sub, last_n)


def fetch_ohlc_batch(
    symbols: list[str],
    days: int = FETCH_DAYS,
    interval: str = "5m",
    last_n: int = CANDLES_NEEDED,
) -> dict[str, list[dict[str, float]]]:
    """
    Single Yahoo request for all symbols (fast). Keys are normalized NSE symbols.
    Missing or failed symbols are omitted; caller may call fetch_ohlc() for retries with fallbacks.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance is required; install with: pip install yfinance")

    unique_nse = list(dict.fromkeys(_normalize_nse(s) for s in symbols if _normalize_nse(s)))
    if not unique_nse:
        return {}

    nse_to_yahoo = {s: _ensure_nse_symbol(s) for s in unique_nse}
    yahoo_list = list(dict.fromkeys(nse_to_yahoo.values()))

    _suppress_yfinance_noise()
    period = f"{days}d"

    try:
        data = yf.download(
            yahoo_list,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            prepost=False,
            threads=True,
            group_by="ticker",
        )
    except Exception as e:
        logger.warning("yfinance batch download failed: %s", e)
        return {}

    if data is None or data.empty:
        return {}

    yahoo_to_nse: dict[str, list[str]] = {}
    for nse, yh in nse_to_yahoo.items():
        yahoo_to_nse.setdefault(yh, []).append(nse)

    out: dict[str, list[dict[str, float]]] = {}
    for yh, nse_list in yahoo_to_nse.items():
        candles = _candles_for_yahoo_ticker(data, yh, last_n)
        if not candles:
            continue
        for nse in nse_list:
            out[nse] = candles

    # Only re-fetch symbols that have alternate Yahoo tickers (batch uses primary .NS only).
    missing_fb = [
        s
        for s in unique_nse
        if (s not in out or len(out.get(s, [])) < 8) and s in YAHOO_FALLBACK_TICKERS
    ]
    for sym in missing_fb:
        try:
            rows = fetch_ohlc(sym, days=days, interval=interval, last_n=last_n)
            if len(rows) >= 8:
                out[sym] = rows
        except Exception:
            pass

    return out
