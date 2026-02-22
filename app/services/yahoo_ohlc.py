"""Fetch OHLC data from Yahoo Finance for NSE symbols."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Number of 5-min candles to use as model input
CANDLES_NEEDED = 40
# Days of history to request (Yahoo 5m usually allows up to 7d)
FETCH_DAYS = 6

# NSE symbol -> Yahoo ticker (without .NS). Use when Yahoo uses a different symbol.
YAHOO_SYMBOL_OVERRIDES = {
    "LTIMINDST": "LTIM",  # LTIMindtree: Yahoo uses LTIM, NSE uses LTIM
}


def _ensure_nse_symbol(symbol: str) -> str:
    """Return Yahoo Finance ticker (e.g. SYMBOL.NS). Uses overrides when Yahoo symbol differs from NSE."""
    s = (symbol or "").strip().upper()
    if not s:
        return s
    base = YAHOO_SYMBOL_OVERRIDES.get(s, s)
    return base if base.endswith(".NS") else f"{base}.NS"


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
        import yfinance as yf
    except ImportError:
        raise RuntimeError("yfinance is required; install with: pip install yfinance")

    # Reduce noisy 404/Failed download messages from yfinance/urllib
    for name in ("yfinance", "urllib3.connectionpool"):
        logging.getLogger(name).setLevel(logging.WARNING)

    ticker = _ensure_nse_symbol(symbol)
    period = f"{days}d"

    try:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            prepost=False,
            threads=False,
        )
    except Exception as e:
        logger.warning("yfinance download failed for %s: %s", ticker, e)
        raise ValueError(f"Failed to fetch data for {symbol}: {e}") from e

    if data is None or data.empty:
        raise ValueError(f"No OHLC data returned for {symbol}")

    # Normalize columns: single ticker -> Open, High, Low, Close; multi -> use first level
    cols = list(data.columns)
    if cols and isinstance(cols[0], tuple):
        # MultiIndex: take (Ticker, Open) -> use second part
        col_map = {}
        for c in cols:
            if isinstance(c, tuple) and len(c) >= 2:
                name = str(c[1]).lower()
                if "open" in name:
                    col_map["open"] = c
                elif "high" in name:
                    col_map["high"] = c
                elif "low" in name:
                    col_map["low"] = c
                elif "close" in name and "adj" not in name:
                    col_map["close"] = c
        o_col = col_map.get("open", cols[0])
        h_col = col_map.get("high", cols[1] if len(cols) > 1 else cols[0])
        l_col = col_map.get("low", cols[2] if len(cols) > 2 else cols[0])
        c_col = col_map.get("close", cols[3] if len(cols) > 3 else cols[0])
    else:
        # Single ticker: Open, High, Low, Close
        o_col = "Open" if "Open" in data.columns else data.columns[0]
        h_col = "High" if "High" in data.columns else data.columns[min(1, len(data.columns) - 1)]
        l_col = "Low" if "Low" in data.columns else data.columns[min(2, len(data.columns) - 1)]
        c_col = "Close" if "Close" in data.columns else data.columns[min(3, len(data.columns) - 1)]

    rows: list[dict[str, float]] = []
    for _, row in data.iterrows():
        rows.append({
            "open": float(row[o_col]),
            "high": float(row[h_col]),
            "low": float(row[l_col]),
            "close": float(row[c_col]),
        })

    if len(rows) < 8:
        raise ValueError(f"Not enough candles for {symbol} (got {len(rows)}, need at least 8)")

    # Return last_n candles (or all if fewer)
    return rows[-last_n:] if len(rows) >= last_n else rows
