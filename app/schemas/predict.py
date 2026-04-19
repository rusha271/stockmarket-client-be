"""Request and response schemas for the prediction endpoint."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictAllRequest(BaseModel):
    """POST /predict/all body. Required non-empty symbols list."""

    model_config = ConfigDict(populate_by_name=True)

    symbols: list[str] = Field(..., min_length=1, description="List of symbols to predict (e.g. AAPL, MSFT)")
    current_prices: dict[str, float] | None = Field(
        None,
        alias="currentPrices",
        description="Optional live last prices by symbol (matches frontend); used when Yahoo OHLC succeeds for alignment",
    )


class PredictAllItem(BaseModel):
    """Single prediction item for /predict/all response."""

    symbol: str = Field(..., description="Symbol")
    predicted_price: float = Field(..., description="Predicted price")


class PredictRequest(BaseModel):
    """POST /predict body. Send currentPrice so direction matches (current − predicted)."""

    model_config = ConfigDict(populate_by_name=True)

    symbol: str = Field(..., min_length=1, max_length=20, description="NSE symbol (e.g. ICICIBANK)")
    current_price: float | None = Field(None, gt=0, alias="currentPrice", description="Optional: live price; direction = (current − predicted) negative→up, positive→down")


class Factors(BaseModel):
    """Factor weights (each 0-1)."""

    technical: float = Field(..., ge=0, le=1)
    fundamental: float = Field(..., ge=0, le=1)
    sentiment: float = Field(..., ge=0, le=1)


class PredictResponse(BaseModel):
    """Prediction response (snake_case for API)."""

    predicted_price: float = Field(..., description="Predicted price")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    direction: Literal["up", "down", "neutral"] = Field(..., description="Price direction")
    timeframe: str = Field(..., description="e.g. 7 days, 1 month")
    reasoning: list[str] = Field(..., description="Short explanation bullets")
    risk_level: Literal["low", "medium", "high"] = Field(..., description="Risk level")
    factors: Factors = Field(..., description="technical, fundamental, sentiment weights")
