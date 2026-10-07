"""
security.py
Password hashing (bcrypt) and JWT creation/verification.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import bcrypt
import jwt
from jwt import PyJWTError

from src import config


def hash_password(plain_password: str) -> str:
    if len(plain_password.encode('utf-8')) > 72: raise ValueError('Password exceeds bcrypt 72-byte limit')
    return bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except ValueError:
        # Malformed hash (e.g. legacy/corrupted record) — treat as invalid, not a crash.
        return False


def create_access_token(data: dict, expires_minutes: Optional[int] = None) -> Tuple[str, int]:
    """Encodes `data` into a signed JWT. Returns (token, expires_in_minutes)."""
    expires_minutes = expires_minutes or config.JWT_EXPIRE_MINUTES
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    to_encode = {**data, "exp": expire}
    token = jwt.encode(to_encode, config.JWT_SECRET_KEY, algorithm=config.JWT_ALGORITHM)
    return token, expires_minutes


def decode_access_token(token: str) -> Optional[dict]:
    """Returns the decoded payload, or None if the token is invalid/expired."""
    try:
        return jwt.decode(token, config.JWT_SECRET_KEY, algorithms=[config.JWT_ALGORITHM])
    except PyJWTError:
        return None
