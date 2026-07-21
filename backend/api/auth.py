import os
from datetime import datetime, timedelta

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db, User

SECRET_KEY = os.getenv("JWT_SECRET", "change-me-in-production-please")
ALGORITHM  = "HS256"
EXPIRE_DAYS = 7

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

router = APIRouter()


# ── Schemas ────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    name: str
    is_admin: bool = False


# ── Helpers ────────────────────────────────────────────────

def _hash(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()

def _verify(pw: str, hashed: str) -> bool:
    return bcrypt.checkpw(pw.encode(), hashed.encode())

def _create_token(user_id: str, email: str) -> str:
    exp = datetime.utcnow() + timedelta(days=EXPIRE_DAYS)
    return jwt.encode({"sub": user_id, "email": email, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

def can_manage_kb(user: User) -> bool:
    """If ADMIN_EMAILS (comma-separated) is set in the environment, only those
    accounts are admins — they may modify the shared KB and view the monitoring
    dashboard; otherwise any signed-in user can (convenient for development,
    lock it down in production). Lives here so both auth responses and the
    KB/monitoring routers share one source of truth."""
    admins = {e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()}
    return not admins or user.email.lower() in admins

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


# ── Endpoints ──────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=req.email, hashed_password=_hash(req.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    token = _create_token(user.id, user.email)
    return TokenResponse(access_token=token, user_id=user.id, email=user.email,
                         name=req.name, is_admin=can_manage_kb(user))


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not _verify(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = _create_token(user.id, user.email)
    return TokenResponse(access_token=token, user_id=user.id, email=user.email,
                         name="", is_admin=can_manage_kb(user))


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {"user_id": current_user.id, "email": current_user.email,
            "is_admin": can_manage_kb(current_user)}
