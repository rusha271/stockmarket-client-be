"""Prediction business logic. Symbol → Yahoo OHLC → XGBoost candlestick model → frontend response."""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Literal

from app.schemas.predict import Factors, PredictResponse
from app.schemas.candlestick import CandlestickPredictResponse

logger = logging.getLogger(__name__)


def direction_from_current_vs_predicted(current_price: float, predicted_price: float) -> Literal["up", "down", "neutral"]:
    """Return direction from current vs predicted price comparison."""
    diff = current_price - predicted_price
    if diff < 0:
        return "up"    # current < predicted → price expected to go up
    if diff > 0:
        return "down"  # current > predicted → price expected to go down
    return "neutral"


def _normalize_symbol(symbol: str) -> str:
    """Normalize symbol to uppercase and strip."""
    return (symbol or "").strip().upper()


def _json_safe_predicted_price(value: float | None) -> float | None:
    """JSON cannot encode NaN/Inf; treat as missing prediction."""
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return round(x, 2)


def _symbol_seed(symbol: str) -> int:
    """Deterministic seed from symbol for reproducible mock output."""
    return int(hashlib.sha256(symbol.encode()).hexdigest()[:8], 16)


def get_prediction(symbol: str) -> PredictResponse:
    """
    Produce a prediction for the given NSE symbol.
    Uses symbol-derived values so output is consistent per symbol; replace this
    with real AI/ML when integrating your model.
    """
    sym = _normalize_symbol(symbol)
    if not sym:
        raise ValueError("Missing or invalid symbol")

    seed = _symbol_seed(sym)

    # Deterministic but varied numeric outputs from seed
    def _pct(lo: float, hi: float) -> float:
        return lo + (seed % 10001) / 10000 * (hi - lo)

    base_price = 100 + (seed % 9000) / 10
    predicted_price = round(base_price * (0.95 + _pct(0, 0.15)), 2)

    confidence = round(0.5 + (seed % 51) / 100, 2)
    confidence = min(1.0, max(0.0, confidence))

    direction_idx = seed % 3
    direction = ("up", "down", "neutral")[direction_idx]

    timeframe_choices = ["7 days", "1 month", "2 weeks", "5 days"]
    timeframe = timeframe_choices[seed % len(timeframe_choices)]

    risk_idx = seed % 3
    risk_level = ("low", "medium", "high")[risk_idx]

    # Dynamic reasoning bullets (no hardcoded example text)
    reasoning_templates = [
        "Price action and support/resistance levels indicate {outlook} near-term.",
        "Volume and momentum metrics align with {direction} bias for this symbol.",
        "Sector and index correlation suggests {risk} volatility in the chosen timeframe.",
        "Historical pattern similarity supports the given {direction} scenario.",
        "Liquidity and spread conditions are favourable for the stated {timeframe} view.",
    ]
    outlook = "strength" if direction == "up" else "weakness" if direction == "down" else "range-bound movement"
    reasoning = [
        t.format(outlook=outlook, direction=direction, risk=risk_level, timeframe=timeframe)
        for t in reasoning_templates[: (3 + seed % 3)]
    ]

    # Factors sum to 1.0, non-negative
    t = _pct(0.2, 0.5)
    f = _pct(0.2, 0.5)
    s = 1.0 - t - f
    if s < 0:
        s, t, f = 0, t / (t + f), f / (t + f)
    factors = Factors(
        technical=round(t, 2),
        fundamental=round(f, 2),
        sentiment=round(max(0, s), 2),
    )
    # Normalize to sum 1.0
    total = factors.technical + factors.fundamental + factors.sentiment
    if total > 0:
        factors = Factors(
            technical=round(factors.technical / total, 2),
            fundamental=round(factors.fundamental / total, 2),
            sentiment=round(factors.sentiment / total, 2),
        )

    return PredictResponse(
        predicted_price=predicted_price,
        confidence=confidence,
        direction=direction,
        timeframe=timeframe,
        reasoning=reasoning,
        risk_level=risk_level,
        factors=factors,
    )


def _candlestick_to_predict_response(
    cr: CandlestickPredictResponse,
    timeframe_str: str | None = None,
    current_price_for_direction: float | None = None,
) -> PredictResponse:
    """Map candlestick model output to frontend PredictResponse."""
    # Use provided current price, or fall back to deriving from regression (predicted vs implied current)
    if current_price_for_direction is not None and current_price_for_direction > 0:
        current_used = current_price_for_direction
        predicted_price = round(current_used * (1 + cr.price_change_pct / 100.0), 2)
    else:
        # No current from request; use model's predicted_price and infer current from it for direction
        predicted_price = cr.predicted_price
        current_used = predicted_price / (1 + cr.price_change_pct / 100.0) if cr.price_change_pct != -100 else predicted_price
    direction = direction_from_current_vs_predicted(current_used, predicted_price)
    if cr.category <= 1:
        risk_level = "low"
    elif cr.category <= 3:
        risk_level = "medium"
    else:
        risk_level = "high"
    reasoning = [
        "Based on 5-min candlestick model.",
        f"Predicted move: {cr.price_change_pct:+.2f}% (category {cr.category}); confidence {cr.probability:.0%}.",
    ]
    from app.services.next_timeframe import get_next_5min_candle_ist
    timeframe = timeframe_str or get_next_5min_candle_ist()
    if not math.isfinite(predicted_price):
        predicted_price = round(current_used, 2)
    conf = cr.probability if math.isfinite(cr.probability) else 0.5
    return PredictResponse(
        predicted_price=predicted_price,
        confidence=conf,
        direction=direction,
        timeframe=timeframe,
        reasoning=reasoning,
        risk_level=risk_level,
        factors=Factors(technical=0.4, fundamental=0.35, sentiment=0.25),
    )


def _predict_from_candles(
    sym: str,
    candles: list[dict],
    clf,
    reg,
    current_price_from_request: float | None,
) -> PredictResponse:
    from app.services.feature_builder import build_features_from_candles
    from app.services.candlestick_service import predict_candlestick, get_num_features
    from app.services.next_timeframe import get_next_5min_candle_ist, is_peak_hours_ist

    if len(candles) < 8:
        raise ValueError(f"Not enough candles: {len(candles)}")

    n = get_num_features(clf) or get_num_features(reg) or 49
    features = build_features_from_candles(candles, n)
    current_close = float(candles[-1]["close"])
    if not math.isfinite(current_close) or current_close <= 0:
        raise ValueError("Invalid close price")

    use_small_bins = not is_peak_hours_ist()
    cr = predict_candlestick(features, current_close, clf, reg, use_5min_bins=use_small_bins)
    current_for_direction = (
        current_price_from_request
        if current_price_from_request is not None and current_price_from_request > 0
        else current_close
    )
    return _candlestick_to_predict_response(
        cr,
        timeframe_str=get_next_5min_candle_ist(),
        current_price_for_direction=current_for_direction,
    )


def get_prediction_from_yahoo(
    symbol: str,
    clf,
    reg,
    current_price_from_request: float | None = None,
    *,
    allow_mock_on_failure: bool = True,
) -> PredictResponse:
    """
    Fetch 6 days OHLC from Yahoo, run candlestick model, return frontend response.
    When current_price_from_request is sent (frontend live price), direction uses (current − predicted).
    On failure: mock get_prediction(symbol) only if allow_mock_on_failure is True.
    """
    sym = _normalize_symbol(symbol)
    if not sym:
        raise ValueError("Missing or invalid symbol")

    try:
        from app.services.yahoo_ohlc import fetch_ohlc

        candles = fetch_ohlc(sym, days=6, interval="5m", last_n=40)
        return _predict_from_candles(sym, candles, clf, reg, current_price_from_request)
    except Exception as e:
        logger.warning("Yahoo/model prediction failed for %s: %s", sym, e)
        if allow_mock_on_failure:
            logger.info("Using mock prediction for %s.", sym)
            return get_prediction(symbol)
        raise


def predict_all_from_yahoo(
    symbols: list[str],
    clf,
    reg,
    current_prices: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    """
    Batch Yahoo OHLC + model for many symbols (one network round-trip when possible).
    Symbols with no usable OHLC get predicted_price None (no mock — avoids bogus prices like TATAMOTORS).
    """
    from app.services.yahoo_ohlc import fetch_ohlc_batch

    prices: dict[str, float] = {}
    if current_prices:
        for k, v in current_prices.items():
            ks = _normalize_symbol(k)
            if ks and v is not None and v > 0:
                prices[ks] = float(v)

    candles_map = fetch_ohlc_batch(symbols, days=6, interval="5m", last_n=40)
    out: list[dict[str, object]] = []

    for raw in symbols:
        sym = _normalize_symbol(raw)
        if not sym:
            continue
        candles = candles_map.get(sym)
        if not candles or len(candles) < 8:
            out.append({"symbol": sym, "predicted_price": None})
            continue
        try:
            resp = _predict_from_candles(sym, candles, clf, reg, prices.get(sym))
            safe = _json_safe_predicted_price(resp.predicted_price)
            out.append({"symbol": sym, "predicted_price": safe})
        except Exception as e:
            logger.warning("Model prediction failed for %s: %s", sym, e)
            out.append({"symbol": sym, "predicted_price": None})

    return out
