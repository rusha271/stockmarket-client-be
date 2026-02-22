"""Request/response schemas for candlestick (XGBoost) prediction."""

from typing import Literal

from pydantic import BaseModel, Field


class CandlestickPredictRequest(BaseModel):
    """
    POST /predict body (Option A: pre-computed features).
    features: first element = timeframe index (0 for 5min), rest = pattern flags (0.0 or 1.0).
    """

    features: list[float] = Field(..., min_length=1, description="Feature vector (e.g. 49 or 53)")
    current_close: float = Field(..., gt=0, description="Last close price for predicted_price")


class CandlestickPredictResponse(BaseModel):
    """Candlestick model output (snake_case for API)."""

    direction: Literal["Bullish", "Bearish"] = Field(..., description="Classifier direction")
    probability: float = Field(..., ge=0, le=1, description="Classifier probability")
    category: int = Field(..., ge=0, le=7, description="Price-change bucket 0-7")
    price_change_pct: float = Field(..., description="Percentage implied by category")
    predicted_price: float = Field(..., description="current_close * (1 + price_change_pct/100)")


# Category → price_change_pct (training / daily-style bins)
CATEGORY_TO_PCT: dict[int, float] = {
    0: -2.5,
    1: -1.5,
    2: -0.75,
    3: -0.25,
    4: 0.25,
    5: 0.75,
    6: 1.5,
    7: 2.5,
}

# Smaller bins for next 5-min candle to minimize price difference (realistic intraday move)
CATEGORY_TO_PCT_5MIN: dict[int, float] = {
    0: -0.25,
    1: -0.15,
    2: -0.075,
    3: -0.025,
    4: 0.025,
    5: 0.075,
    6: 0.15,
    7: 0.25,
}
