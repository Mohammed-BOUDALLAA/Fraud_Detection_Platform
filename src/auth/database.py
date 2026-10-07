"""
database.py
SQLAlchemy engine/session setup for authentication & user management.

Defaults to a local SQLite file (data/app.db) so the whole platform
still runs with zero external infrastructure out of the box. Point
DATABASE_URL at Postgres/MySQL in production via an environment
variable — nothing else in this module needs to change.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from src import config

_connect_args = {"check_same_thread": False} if config.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(config.DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency — yields a DB session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Creates all auth tables if they don't exist yet. Safe to call repeatedly."""
    from src.auth import models  # noqa: F401 — ensures models are registered on Base

    Base.metadata.create_all(bind=engine)
