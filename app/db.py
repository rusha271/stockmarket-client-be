"""Raw SQL database layer (MySQL). No ORM, no Pydantic."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import pymysql
from pymysql.cursors import DictCursor

from app.config import Settings

logger = logging.getLogger(__name__)
_settings = Settings()


def get_connection():
    """Return a new DB connection. Caller must close it."""
    if _settings.database_url:
        # Parse mysql://user:pass@host:port/dbname
        url = _settings.database_url
        if url.startswith("mysql://"):
            url = url[8:]
        parts = url.split("@", 1)
        user_pass = parts[0].split(":", 1)
        host_db = parts[1].split("/", 1)
        host_port = host_db[0].rsplit(":", 1)
        return pymysql.connect(
            host=host_port[0],
            port=int(host_port[1]) if len(host_port) > 1 else 3306,
            user=user_pass[0],
            password=user_pass[1] if len(user_pass) > 1 else "",
            database=host_db[1] if len(host_db) > 1 else "be_stock",
            cursorclass=DictCursor,
        )
    return pymysql.connect(
        host=_settings.db_host,
        port=_settings.db_port,
        user=_settings.db_user,
        password=_settings.db_password,
        database=_settings.db_name,
        cursorclass=DictCursor,
    )


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
    with get_cursor(commit=commit) as cur:
        cur.execute(sql, params or ())
        if cur.description:
            return list(cur.fetchall())
    return []


def init_auth_table():
    """Create users table if not exists (raw SQL, MySQL)."""
    sql = """
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        contact_number VARCHAR(20) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    execute(sql, commit=True)
    logger.info("Auth table users ready.")


def insert_user(email: str, password_hash: str, contact_number: str) -> int | None:
    """Insert user; return id or None on duplicate email."""
    sql = """
    INSERT INTO users (email, password_hash, contact_number)
    VALUES (%s, %s, %s);
    """
    with get_cursor(commit=True) as cur:
        try:
            cur.execute(sql, (email.strip().lower(), password_hash, contact_number.strip()))
            return cur.lastrowid
        except pymysql.IntegrityError:
            return None


def get_user_by_email(email: str) -> dict[str, Any] | None:
    """Fetch user by email; return dict with id, email, password_hash, contact_number, created_at or None."""
    sql = "SELECT id, email, password_hash, contact_number, created_at FROM users WHERE email = %s;"
    rows = execute(sql, (email.strip().lower(),))
    return rows[0] if rows else None
