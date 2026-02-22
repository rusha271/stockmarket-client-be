"""Build one row of model features from OHLC candles (5-min). Matches trainer pattern order when available."""

from __future__ import annotations

from typing import Any


def _body(c: dict[str, float]) -> float:
    return abs(c["close"] - c["open"])


def _range_size(c: dict[str, float]) -> float:
    r = c["high"] - c["low"]
    return r if r > 0 else 1e-8


def _upper_wick(c: dict[str, float]) -> float:
    return c["high"] - max(c["open"], c["close"])


def _lower_wick(c: dict[str, float]) -> float:
    return min(c["open"], c["close"]) - c["low"]


def _is_doji(c: dict[str, float], body_ratio: float = 0.1) -> float:
    r = _range_size(c)
    return 1.0 if _body(c) <= r * body_ratio else 0.0


def _is_hammer(c: dict[str, float]) -> float:
    """Small body at top, long lower wick."""
    r = _range_size(c)
    b = _body(c)
    lw = _lower_wick(c)
    uw = _upper_wick(c)
    if r < 1e-8:
        return 0.0
    return 1.0 if (b <= r * 0.35 and lw >= r * 0.6 and uw <= r * 0.2) else 0.0


def _is_inverted_hammer(c: dict[str, float]) -> float:
    """Small body at bottom, long upper wick."""
    r = _range_size(c)
    b = _body(c)
    lw = _lower_wick(c)
    uw = _upper_wick(c)
    if r < 1e-8:
        return 0.0
    return 1.0 if (b <= r * 0.35 and uw >= r * 0.6 and lw <= r * 0.2) else 0.0


def _is_bullish_engulfing(candles: list[dict[str, float]], i: int) -> float:
    if i < 1:
        return 0.0
    prev, cur = candles[i - 1], candles[i]
    if prev["close"] >= prev["open"]:
        return 0.0
    return 1.0 if (cur["open"] < prev["close"] and cur["close"] > prev["open"] and cur["close"] > cur["open"]) else 0.0


def _is_bearish_engulfing(candles: list[dict[str, float]], i: int) -> float:
    if i < 1:
        return 0.0
    prev, cur = candles[i - 1], candles[i]
    if prev["close"] <= prev["open"]:
        return 0.0
    return 1.0 if (cur["open"] > prev["close"] and cur["close"] < prev["open"] and cur["close"] < cur["open"]) else 0.0


def _pattern_flags_from_candles(candles: list[dict[str, float]]) -> list[float]:
    """Compute pattern flags from last candles. Order should match Colab _define_patterns()."""
    if not candles:
        return []
    last = candles[-1]
    flags: list[float] = []
    # Doji
    flags.append(_is_doji(last))
    # Hammer
    flags.append(_is_hammer(last))
    # Inverted Hammer
    flags.append(_is_inverted_hammer(last))
    # Engulfing (use last two candles)
    flags.append(_is_bullish_engulfing(candles, len(candles) - 1))
    flags.append(_is_bearish_engulfing(candles, len(candles) - 1))
    # Additional common patterns (placeholders to reach ~49–53 total; replace with Colab order if available)
    for _ in range(44):  # 5 + 44 = 49 pattern flags; adjust if model expects 52
        flags.append(0.0)
    return flags


def build_features_from_candles(candles: list[dict[str, float]], n_features: int) -> list[float]:
    """
    Build one feature row for the 5-min model: [timeframe_index, ...pattern_flags].
    timeframe_index = 0 for 5min. Remaining n_features-1 are pattern flags (0.0 or 1.0).
    """
    if n_features < 2:
        return [0.0] * max(1, n_features)
    pattern_flags = _pattern_flags_from_candles(candles)
    # Pad or slice to (n_features - 1)
    need = n_features - 1
    if len(pattern_flags) >= need:
        pattern_flags = pattern_flags[:need]
    else:
        pattern_flags = pattern_flags + [0.0] * (need - len(pattern_flags))
    return [0.0] + pattern_flags  # 0 = 5min timeframe
