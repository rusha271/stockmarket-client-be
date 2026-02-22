"""API route handlers."""

from fastapi import APIRouter, Request

from app.schemas.predict import PredictRequest, PredictResponse
from app.schemas.candlestick import CandlestickPredictRequest, CandlestickPredictResponse
from app.services.prediction_service import get_prediction
from app.services.candlestick_service import predict_candlestick
from app.core.exceptions import invalid_symbol_error, server_error

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(request: Request, body: PredictRequest) -> PredictResponse:
    """
    POST /predict — AI stock prediction by NSE symbol (for frontend).
    Fetches 6 days OHLC from Yahoo Finance, uses last 40 5-min candles in the model, returns prediction.
    Request body: { "symbol": "<NSE_SYMBOL>" }  e.g. { "symbol": "ASIANPAINT" }
    """
    symbol = (body.symbol or "").strip().upper()
    if not symbol:
        raise invalid_symbol_error("Missing or invalid symbol")

    clf = getattr(request.app.state, "clf", None)
    reg = getattr(request.app.state, "reg", None)
    if clf is not None and reg is not None:
        from app.services.prediction_service import get_prediction_from_yahoo
        return get_prediction_from_yahoo(symbol, clf, reg, current_price_from_request=body.current_price)

    return get_prediction(symbol)


@router.post("/predict/candlestick", response_model=CandlestickPredictResponse)
def predict_candlestick_route(request: Request, body: CandlestickPredictRequest) -> CandlestickPredictResponse:
    """
    POST /predict/candlestick — XGBoost candlestick models (5min).
    Request body: { "features": [0, 0, 1, ...], "current_close": 22015.0 }
    """
    clf = getattr(request.app.state, "clf", None)
    reg = getattr(request.app.state, "reg", None)
    if clf is None or reg is None:
        raise server_error("Models not loaded")

    try:
        return predict_candlestick(
            features=body.features,
            current_close=body.current_close,
            clf=clf,
            reg=reg,
        )
    except Exception as e:
        raise server_error(f"Prediction failed: {str(e)}")
