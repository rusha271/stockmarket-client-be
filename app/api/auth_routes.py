"""Auth API: signup (email, password, contact number) and login (email, password). Raw SQL, no Pydantic."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.db import get_user_by_email, insert_user, init_auth_table
from app.auth_utils import hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


def _require(body: dict, *keys: str) -> list:
    missing = [k for k in keys if not (body.get(k) and str(body.get(k)).strip())]
    if missing:
        raise StarletteHTTPException(400, detail=f"Missing or empty: {', '.join(missing)}")
    return [str(body[k]).strip() for k in keys]


@router.post("/signup")
async def signup(request: Request):
    """
    Sign up: email, password, contact_number.
    After signup you can login with email and password.
    """
    try:
        body = await request.json()
    except Exception:
        raise StarletteHTTPException(400, detail="Invalid JSON body")
    email, password, contact_number = _require(body, "email", "password", "contact_number")
    email = email.lower()
    if len(password) < 6:
        raise StarletteHTTPException(400, detail="Password must be at least 6 characters")
    password_hash = hash_password(password)
    user_id = insert_user(email, password_hash, contact_number)
    if user_id is None:
        raise StarletteHTTPException(409, detail="Email already registered")
    return JSONResponse(
        status_code=201,
        content={"message": "Signed up successfully", "user_id": user_id, "email": email},
    )


@router.post("/login")
async def login(request: Request):
    """Login with email and password."""
    try:
        body = await request.json()
    except Exception:
        raise StarletteHTTPException(400, detail="Invalid JSON body")
    email, password = _require(body, "email", "password")
    email = email.lower()
    user = get_user_by_email(email)
    if not user:
        raise StarletteHTTPException(401, detail="Invalid email or password")
    if not verify_password(password, user["password_hash"]):
        raise StarletteHTTPException(401, detail="Invalid email or password")
    return {
        "message": "Login successful",
        "user_id": user["id"],
        "email": user["email"],
        "contact_number": user["contact_number"],
    }
