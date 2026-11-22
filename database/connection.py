"""Database connection and session management for CloudDash."""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Base class for declarative ORM models
Base = declarative_base()

def get_database_url() -> str:
    """Resolve database URL from environment or default to local SQLite."""
    url = os.getenv("DATABASE_URL", "sqlite:///clouddash.db")
    # Heroku/Render legacy postgres prefix fix for SQLAlchemy
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url

_engine = None
_SessionFactory = None

def get_engine():
    """Get or create singleton SQLAlchemy engine."""
    global _engine
    if _engine is None:
        url = get_database_url()
        connect_args = {}
        if url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}
        _engine = create_engine(url, connect_args=connect_args)
    return _engine

def get_session_factory():
    """Get or create sessionmaker factory."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)
    return _SessionFactory

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Provide a transactional database session."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def init_db():
    """Initialize database tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
