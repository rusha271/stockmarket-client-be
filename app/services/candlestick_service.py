"""Candlestick XGBoost prediction: classifier (Bullish/Bearish) + regression (price-change category)."""

import numpy as np
import xgboost as xgb

from app.schemas.candlestick import (
    CATEGORY_TO_PCT,
    CATEGORY_TO_PCT_5MIN,
    CandlestickPredictResponse,
)


def get_num_features(booster: xgb.Booster) -> int | None:
    """Return number of features expected by the booster, or None if unknown."""
    for name in ("num_features", "num_feature"):
        attr = getattr(booster, name, None)
        if attr is None:
            continue
        try:
            n = int(attr() if callable(attr) else attr)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
    return None


def _prepare_features(features: list[float], n: int) -> np.ndarray:
    """Return shape (1, n): slice or zero-pad to length n."""
    a = np.array(features, dtype=np.float32)
    if len(a) >= n:
        return a[:n].reshape(1, -1)
    pad = np.zeros(n - len(a), dtype=np.float32)
    return np.concatenate([a, pad]).reshape(1, -1)


def predict_candlestick(
    features: list[float],
    current_close: float,
    clf: xgb.Booster,
    reg: xgb.Booster,
    *,
    use_5min_bins: bool = True,
) -> CandlestickPredictResponse:
    """
    Run classifier and regression on one feature row; map category to price_change_pct
    and compute predicted_price. When use_5min_bins=True (default), uses smaller
    percentage bins to minimize price difference for next 5-min prediction.
    """
    n = get_num_features(clf) or get_num_features(reg) or len(features)
    row = _prepare_features(features, n)
    d = xgb.DMatrix(row)

    prob = float(clf.predict(d)[0])
    if not (0 <= prob <= 1):
        try:
            import math
            prob = 1.0 / (1.0 + math.exp(-prob))
        except Exception:
            prob = 0.5
    probability = max(0.0, min(1.0, prob))
    direction = "Bullish" if probability > 0.5 else "Bearish"

    cat_raw = reg.predict(d)[0]
    category = int(round(float(cat_raw)))
    category = max(0, min(7, category))
    pct_map = CATEGORY_TO_PCT_5MIN if use_5min_bins else CATEGORY_TO_PCT
    price_change_pct = pct_map[category]

    predicted_price = round(current_close * (1 + price_change_pct / 100.0), 2)

    return CandlestickPredictResponse(
        direction=direction,
        probability=round(probability, 4),
        category=category,
        price_change_pct=price_change_pct,
        predicted_price=predicted_price,
    )
