"""
schemas.py
Pydantic request/response models for the auth & user-management API.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator

from src.auth.models import Role


class LoginRequest(BaseModel):
    username: str
    password: str
    remember_me: bool = False


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    role: Role
    username: str


class PasswordValidated(BaseModel):
    @field_validator('password', check_fields=False)
    @classmethod
    def bcrypt_length(cls, value):
        if value is not None and len(value.encode('utf-8')) > 72:
            raise ValueError('Password must not exceed 72 UTF-8 bytes')
        return value


class UserCreate(PasswordValidated):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8)
    role: Role = Role.ANALYST


class UserUpdate(PasswordValidated):
    role: Optional[Role] = None
    is_active: Optional[bool] = None
    password: Optional[str] = Field(default=None, min_length=8)


class UserOut(BaseModel):
    id: int
    username: str
    role: Role
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class AuditLogOut(BaseModel):
    id: int
    timestamp: datetime
    username: str
    action: str
    detail: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)
