"""
auth.py — Authentication utilities for the Sinhala Letter RAG System.

Handles:
  - SQLite user database (users table)
  - Password hashing with bcrypt
  - JWT token creation and verification
"""

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from jose import JWTError, jwt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-secret-in-production-env")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")


# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the users table if it doesn't exist."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT    UNIQUE NOT NULL,
                hashed_password TEXT    NOT NULL,
                full_name       TEXT    NOT NULL,
                title           TEXT,
                address_line1   TEXT,
                address_line2   TEXT,
                phone           TEXT,
                created_at      TEXT    DEFAULT (datetime('now'))
            )
        """)
        conn.commit()


# ---------------------------------------------------------------------------
# Password utils
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

def create_user(username: str, password: str, full_name: str,
                title: str = "", address_line1: str = "",
                address_line2: str = "", phone: str = "") -> dict:
    hashed = hash_password(password)
    with get_db_connection() as conn:
        conn.execute(
            """INSERT INTO users
               (username, hashed_password, full_name, title, address_line1, address_line2, phone)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (username, hashed, full_name, title, address_line1, address_line2, phone),
        )
        conn.commit()
    return get_user_by_username(username)


def get_user_by_username(username: str) -> Optional[dict]:
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


def update_user_profile(username: str, full_name: str, title: str,
                        address_line1: str, address_line2: str, phone: str) -> dict:
    with get_db_connection() as conn:
        conn.execute(
            """UPDATE users
               SET full_name=?, title=?, address_line1=?, address_line2=?, phone=?
               WHERE username=?""",
            (full_name, title, address_line1, address_line2, phone, username),
        )
        conn.commit()
    return get_user_by_username(username)


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


# ---------------------------------------------------------------------------
# JWT utils
# ---------------------------------------------------------------------------

def create_access_token(username: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[str]:
    """Returns the username from the token, or None if invalid/expired."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None
