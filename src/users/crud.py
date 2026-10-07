"""
crud.py
Database operations for user management and the security audit trail.
Kept separate from the API routing (auth_router.py) so the same
functions can be reused by init_admin.py and, later, any other tooling.
"""

from typing import List, Optional

from sqlalchemy.orm import Session

from src.auth.models import User, Role, AuditLog
from src.auth.security import hash_password


# ── Users ────────────────────────────────────────────────────────────────
def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def list_users(db: Session) -> List[User]:
    return db.query(User).order_by(User.id).all()


def create_user(db: Session, username: str, password: str, role: Role = Role.ANALYST) -> User:
    user = User(username=username, hashed_password=hash_password(password), role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_user(
    db: Session,
    user: User,
    *,
    role: Optional[Role] = None,
    is_active: Optional[bool] = None,
    password: Optional[str] = None,
) -> User:
    if role is not None:
        user.role = role
    if is_active is not None:
        user.is_active = is_active
    if password is not None:
        user.hashed_password = hash_password(password)
    db.commit()
    db.refresh(user)
    return user


def delete_user(db: Session, user: User) -> None:
    db.delete(user)
    db.commit()


# ── Audit trail ──────────────────────────────────────────────────────────
def log_audit(db: Session, username: str, action: str, detail: str = "") -> None:
    """
    Records a security-relevant event. Never raises — a logging failure
    must never block the action it's trying to record (login, save,
    user management, ...).
    """
    try:
        db.add(AuditLog(username=username, action=action, detail=detail))
        db.commit()
    except Exception:
        db.rollback()


def list_audit_logs(db: Session, limit: int = 200) -> List[AuditLog]:
    return db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit).all()
