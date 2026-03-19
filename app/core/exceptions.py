"""Custom exceptions for consistent error responses."""

from fastapi import HTTPException
from fastapi.responses import JSONResponse


def invalid_symbol_error(detail: str = "Missing or invalid symbol") -> HTTPException:
    """Return 400 for invalid or missing symbol."""
    return HTTPException(status_code=400, detail=detail)


def server_error(detail: str) -> HTTPException:
    """Return 500 for server/processing errors."""
    return HTTPException(status_code=500, detail=detail)


def error_response(message: str, status_code: int = 400) -> JSONResponse:
    """Return JSON error response with { \"error\": \"message\" } shape."""
    return JSONResponse(status_code=status_code, content={"error": message})
