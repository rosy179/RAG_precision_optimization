import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, String, Text, DateTime, ForeignKey, Integer, text
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship

DB_PATH = Path(__file__).parent.parent / "data" / "webapp.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id             = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email          = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    created_at     = Column(DateTime, default=datetime.utcnow)
    sessions       = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    documents      = relationship("Document", back_populates="user", cascade="all, delete-orphan")


class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id    = Column(String, ForeignKey("users.id"), nullable=False)
    title      = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    user       = relationship("User", back_populates="sessions")
    messages   = relationship("Message", back_populates="session", cascade="all, delete-orphan",
                              order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"
    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id   = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role         = Column(String, nullable=False)  # "user" | "assistant"
    content      = Column(Text, nullable=False)
    sources_json = Column(Text, default="[]")
    # Names of documents the user attached alongside this message (user role only).
    attachments_json = Column(Text, default="[]")
    # User rating on assistant messages: "up" | "down" | NULL (no rating).
    feedback     = Column(String, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow)
    session      = relationship("ChatSession", back_populates="messages")


class Document(Base):
    __tablename__ = "documents"
    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id     = Column(String, ForeignKey("users.id"), nullable=False)
    # Documents are scoped to the conversation they were uploaded in;
    # NULL only for legacy rows created before session scoping.
    session_id  = Column(String, ForeignKey("chat_sessions.id"), nullable=True)
    name        = Column(String, nullable=False)
    doc_type    = Column(String, nullable=False)  # "pdf" | "url" | "image"
    chunk_count = Column(Integer, default=0)
    created_at  = Column(DateTime, default=datetime.utcnow)
    user        = relationship("User", back_populates="documents")


def create_tables():
    Base.metadata.create_all(bind=engine)
    # Lightweight migration: create_all never alters existing tables.
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(documents)"))]
        if "session_id" not in cols:
            conn.execute(text("ALTER TABLE documents ADD COLUMN session_id VARCHAR"))
            conn.commit()
        msg_cols = [row[1] for row in conn.execute(text("PRAGMA table_info(messages)"))]
        if "attachments_json" not in msg_cols:
            conn.execute(text("ALTER TABLE messages ADD COLUMN attachments_json TEXT DEFAULT '[]'"))
            conn.commit()
        if "feedback" not in msg_cols:
            conn.execute(text("ALTER TABLE messages ADD COLUMN feedback VARCHAR"))
            conn.commit()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
