"""API route handlers."""

from fastapi import APIRouter, Request

from app.schemas.predict import PredictRequest, PredictResponse, PredictAllRequest
from app.schemas.candlestick import CandlestickPredictRequest, CandlestickPredictResponse
from app.services.prediction_service import get_prediction, get_prediction_from_yahoo
from app.services.candlestick_service import predict_candlestick
from app.core.exceptions import invalid_symbol_error, server_error, error_response

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


@router.post("/predict/all")
def predict_all(request: Request, body: PredictAllRequest):
    """
    POST /predict/all — Batch predictions for multiple symbols.
    Request body: { "symbols": ["AAPL", "MSFT", "TSLA"] }
    Response: [{ "symbol": "AAPL", "predicted_price": 195.2 }, ...]
    """
    # Validate: symbols required, non-empty; each symbol non-empty after trim
    symbols_raw = body.symbols
    if not symbols_raw:
        return error_response("symbols is required and must be non-empty", status_code=422)

    normalized: list[str] = []
    for s in symbols_raw:
        if not isinstance(s, str):
            return error_response("Each symbol must be a non-empty string", status_code=422)
        sym = (s or "").strip().upper()
        if not sym:
            return error_response("Symbols cannot contain empty or whitespace-only values", status_code=422)
        normalized.append(sym)

    clf = getattr(request.app.state, "clf", None)
    reg = getattr(request.app.state, "reg", None)
    use_yahoo = clf is not None and reg is not None

    predictions: list[dict] = []
    for symbol in normalized:
        try:
            if use_yahoo:
                resp = get_prediction_from_yahoo(symbol, clf, reg, current_price_from_request=None)
            else:
                resp = get_prediction(symbol)
            predictions.append({"symbol": symbol, "predicted_price": resp.predicted_price})
        except Exception as e:
            # On per-symbol failure: include symbol with null predicted_price (frontend expects null for invalid/missing)
            predictions.append({"symbol": symbol, "predicted_price": None})

    return predictions


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
