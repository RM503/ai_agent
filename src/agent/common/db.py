"""
Script that initializes PostgreSQL db for persisting
conversations.
"""
from __future__ import annotations 

import os
from collections.abc import Generator

from sqlmodel import Session, create_engine

POSTGRESQL_CONNECTION = os.getenv("POSTGRESQL_CONNECTION")
engine = create_engine(POSTGRESQL_CONNECTION)

def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        try:
            yield session 
            session.commit()
        except Exception:
            session.rollback()
            raise