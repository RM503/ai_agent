from datetime import datetime, UTC
from typing import Optional
from uuid import UUID

from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

class ChatSession(SQLModel, table=True):
    __tablename__ = "chat_sessions"
    __table_args__ = {"extend_existing": True}

    session_id: Optional[UUID] = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"server_default": "gen_random_uuid()"}
    )
    user_id: Optional[UUID] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=datetime.now(UTC), nullable=False)
    updated_at: datetime = Field(default_factory=datetime.now(UTC), nullable=False)
    last_message_at: Optional[datetime] = Field(default=None)
    session_metadata: dict = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False),
    )