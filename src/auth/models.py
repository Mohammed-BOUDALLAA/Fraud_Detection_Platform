"""
models.py
SQLAlchemy ORM models for authentication, user management, and the
security audit trail.
"""

import enum
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text
from sqlalchemy import Enum as SAEnum

from src.auth.database import Base


class Role(str, enum.Enum):
    ADMIN = "admin"
    ANALYST = "analyst"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    hashed_password = Column(String(128), nullable=False)
    role = Column(SAEnum(Role), nullable=False, default=Role.ANALYST)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_login = Column(DateTime, nullable=True)


class AuditLog(Base):
    """
    Security-relevant event trail — separate from the plain-text
    component logs in logs/*.log. Powers the dashboard's "📝 Journal
    d'audit" tab (admin only): logins, model saves, and every user-
    management action, each attributed to a specific user.
    """

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    username = Column(String(64), nullable=False)
    action = Column(String(64), nullable=False)
    detail = Column(Text, nullable=True)
