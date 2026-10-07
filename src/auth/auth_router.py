"""
auth_router.py
/auth/login, /auth/me, admin-only user-management endpoints, and the
audit-log listing endpoint. Mounted into the main FastAPI app in
src/serving/api.py.
"""

from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.auth.database import get_db
from src.auth.dependencies import get_current_user, require_admin
from src.auth.models import User
from src.auth.schemas import (
    LoginRequest,
    Token,
    UserCreate,
    UserOut,
    UserUpdate,
    AuditLogOut,
)
from src.auth.security import create_access_token, verify_password
from src.users import crud
from src import config

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=Token)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = crud.get_user_by_username(db, payload.username)

    if user is None or not verify_password(payload.password, user.hashed_password):
        # Same error for "no such user" and "wrong password" — never reveal
        # which one it was, that would let an attacker enumerate usernames.
        crud.log_audit(db, payload.username, "login_failed")
        raise HTTPException(status_code=401, detail="Nom d'utilisateur ou mot de passe invalide")

    if not user.is_active:
        crud.log_audit(db, user.username, "login_blocked_inactive")
        raise HTTPException(status_code=403, detail="Compte désactivé — contactez un administrateur")

    expires_minutes = config.JWT_REMEMBER_ME_MINUTES if payload.remember_me else config.JWT_EXPIRE_MINUTES
    token, expires_in = create_access_token(
        {"sub": user.username, "role": user.role.value}, expires_minutes=expires_minutes
    )

    user.last_login = datetime.now(timezone.utc)
    db.commit()
    crud.log_audit(db, user.username, "login_success", detail=f"remember_me={payload.remember_me}")

    return Token(
        access_token=token, expires_in_minutes=expires_in, role=user.role, username=user.username
    )


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return current_user


# ── User management (admin only) ───────────────────────────────────────
@router.get("/users", response_model=List[UserOut], dependencies=[Depends(require_admin)])
def get_users(db: Session = Depends(get_db)):
    return crud.list_users(db)


@router.post("/users", response_model=UserOut, dependencies=[Depends(require_admin)])
def add_user(
    payload: UserCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    if crud.get_user_by_username(db, payload.username):
        raise HTTPException(status_code=409, detail="Ce nom d'utilisateur existe déjà")
    user = crud.create_user(db, payload.username, payload.password, payload.role)
    crud.log_audit(
        db, current_user.username, "user_created", detail=f"{payload.username} ({payload.role.value})"
    )
    return user


@router.patch("/users/{user_id}", response_model=UserOut, dependencies=[Depends(require_admin)])
def edit_user(
    user_id: int,
    payload: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    user = crud.get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    if user.id == current_user.id and (payload.is_active is False or (payload.role is not None and payload.role != current_user.role)):
        raise HTTPException(status_code=400, detail="Vous ne pouvez pas désactiver votre propre compte")

    updated = crud.update_user(
        db, user, role=payload.role, is_active=payload.is_active, password=payload.password
    )
    crud.log_audit(db, current_user.username, "user_updated", detail=f"user_id={user_id}")
    return updated


@router.delete("/users/{user_id}", dependencies=[Depends(require_admin)])
def remove_user(
    user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
):
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Vous ne pouvez pas supprimer votre propre compte")
    user = crud.get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")

    crud.delete_user(db, user)
    crud.log_audit(db, current_user.username, "user_deleted", detail=f"user_id={user_id}")
    return {"deleted": True}


# ── Audit trail (admin only) ─────────────────────────────────────────────
@router.get("/audit-logs", response_model=List[AuditLogOut], dependencies=[Depends(require_admin)])
def get_audit_logs(limit: int = Query(default=200, ge=1, le=1000), db: Session = Depends(get_db)):
    return crud.list_audit_logs(db, limit=limit)
