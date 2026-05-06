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

DB_PATH = os.getenv(
    "USERS_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db"),
)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


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


# ---------------------------------------------------------------------------
# Feedback database
# ---------------------------------------------------------------------------

def init_feedback_db():
    """Create feedback tables if they don't exist."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS letter_ratings (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id           INTEGER,
                username          TEXT,
                timestamp         TEXT DEFAULT (datetime('now')),
                original_prompt   TEXT,
                letter_content    TEXT,
                letter_category   TEXT,
                quality_overall   INTEGER,
                quality_match     INTEGER,
                quality_language  INTEGER,
                quality_structure INTEGER,
                comments          TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_feedback (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id                 INTEGER,
                username                TEXT,
                timestamp               TEXT DEFAULT (datetime('now')),
                ease_of_use             INTEGER,
                ease_of_describing      INTEGER,
                gap_questions_helpful   INTEGER,
                confidence_in_output    INTEGER,
                would_use_again         INTEGER,
                liked_most              TEXT,
                needs_improvement       TEXT,
                issues_faced            TEXT
            )
        """)
        conn.commit()


def save_letter_rating(user_id: int, username: str, original_prompt: str,
                       letter_content: str, letter_category: str,
                       quality_overall: int, quality_match: int,
                       quality_language: int, quality_structure: int,
                       comments: str = "") -> int:
    with get_db_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO letter_ratings
               (user_id, username, original_prompt, letter_content, letter_category,
                quality_overall, quality_match, quality_language, quality_structure, comments)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, username, original_prompt, letter_content, letter_category,
             quality_overall, quality_match, quality_language, quality_structure, comments),
        )
        conn.commit()
        return cursor.lastrowid


def save_system_feedback(user_id: int, username: str,
                         ease_of_use: int, ease_of_describing: int,
                         gap_questions_helpful: int, confidence_in_output: int,
                         would_use_again: int,
                         liked_most: str = "", needs_improvement: str = "",
                         issues_faced: str = "") -> int:
    with get_db_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO system_feedback
               (user_id, username, ease_of_use, ease_of_describing,
                gap_questions_helpful, confidence_in_output, would_use_again,
                liked_most, needs_improvement, issues_faced)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, username, ease_of_use, ease_of_describing,
             gap_questions_helpful, confidence_in_output, would_use_again,
             liked_most, needs_improvement, issues_faced),
        )
        conn.commit()
        return cursor.lastrowid


def get_all_letter_ratings() -> list:
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM letter_ratings ORDER BY timestamp DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_system_feedback() -> list:
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM system_feedback ORDER BY timestamp DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_feedback_stats() -> dict:
    with get_db_connection() as conn:
        lr = conn.execute("""
            SELECT
                COUNT(*) as total,
                ROUND(AVG(quality_overall), 2)   as avg_overall,
                ROUND(AVG(quality_match), 2)      as avg_match,
                ROUND(AVG(quality_language), 2)   as avg_language,
                ROUND(AVG(quality_structure), 2)  as avg_structure
            FROM letter_ratings
        """).fetchone()
        sf = conn.execute("""
            SELECT
                COUNT(*) as total,
                ROUND(AVG(ease_of_use), 2)           as avg_ease_of_use,
                ROUND(AVG(ease_of_describing), 2)    as avg_ease_of_describing,
                ROUND(AVG(gap_questions_helpful), 2) as avg_gap_questions,
                ROUND(AVG(confidence_in_output), 2)  as avg_confidence,
                ROUND(AVG(would_use_again), 2)       as avg_would_use_again
            FROM system_feedback
        """).fetchone()
        users = conn.execute("SELECT COUNT(*) as total FROM users").fetchone()
    return {
        "users": {"total": users["total"]},
        "letter_ratings": dict(lr),
        "system_feedback": dict(sf),
    }
