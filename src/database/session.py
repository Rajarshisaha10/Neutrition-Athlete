from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base


def database_url() -> str:
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "diagnosis_cases.db")
    return os.getenv("DATABASE_URL", f"sqlite:///{db_path}")


def _engine():
    url = database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)


ENGINE = _engine()
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=ENGINE)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
