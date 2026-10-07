"""
dependencies.py
FastAPI dependencies for authenticating requests and enforcing roles.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session

from src.auth.database import get_db
from src.auth.models import User, Role
from src.auth.security import decode_access_token

# tokenUrl is only used to populate the OpenAPI docs' "Authorize" button;
# login itself is a plain JSON POST (see auth_router.py), not an OAuth2 form.
oauth2_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    """Resolves the JWT bearer token into an active User, or raises 401."""
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Impossible de valider les identifiants",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if token is None:
        raise credentials_error

    payload = decode_access_token(token.credentials)
    if payload is None or "sub" not in payload:
        raise credentials_error

    user = db.query(User).filter(User.username == payload["sub"]).first()
    if user is None or not user.is_active:
        raise credentials_error

    return user


def require_roles(*allowed_roles: Role):
    """Dependency factory — raises 403 unless current_user.role is allowed."""

    def _checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Rôle requis : {', '.join(r.value for r in allowed_roles)}",
            )
        return current_user

    return _checker


require_admin = require_roles(Role.ADMIN)
require_any_role = require_roles(Role.ADMIN, Role.ANALYST)
