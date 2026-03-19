"""Raw SQL database layer (MySQL). No ORM, no Pydantic."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

# import pymysql
# from pymysql.cursors import DictCursor

from app.config import Settings

logger = logging.getLogger(__name__)
_settings = Settings()


def get_connection():
    """Return a new DB connection. Caller must close it."""
    raise RuntimeError("Database is temporarily disabled (pymysql commented out).")


@contextmanager
def get_cursor(commit: bool = False):
    """Context manager: connection + cursor. Optionally commit on success."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        try:
            yield cur
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
    finally:
        conn.close()


def execute(sql: str, params: tuple | dict | None = None, commit: bool = False) -> list[dict[str, Any]]:
    """Execute SQL and return rows as list of dicts. Use for SELECT; for INSERT/UPDATE set commit=True."""
    _ = (sql, params, commit)
    return []


def init_auth_table():
    """Create users table if not exists (raw SQL, MySQL)."""
    logger.info("Auth table init skipped: database disabled.")


def insert_user(email: str, password_hash: str, contact_number: str) -> int | None:
    """Insert user; return id or None on duplicate email."""
    _ = (email, password_hash, contact_number)
    return None


def get_user_by_email(email: str) -> dict[str, Any] | None:
    """Fetch user by email; return dict with id, email, password_hash, contact_number, created_at or None."""
    _ = email
    return None
