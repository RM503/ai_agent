from datetime import datetime, UTC
from typing import Optional
from uuid import UUID

from sqlalchemy import Column, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel

class ChatMessage(SQLModel, table=True):
    __tablename__ = "chat_messages"
    __table_args__ = (
        UniqueConstraint("session_id", "message_index"),
        {"extend_existing": True}
    )

    message_id: Optional[uuid.UUID] = Field(
        default=None,
        primary_key=True,
        sa_column_kwargs={"server_default": "gen_random_uuid()"}
    )
    session_id: uuid.UUID = Field(foreign_key="chat_sessions.session_id")
    message_index: int = Field(nullable=False)
    message_role: str = Field(nullable=False)
    message_content: str = Field(nullable=False)
    created_at: datetime = Field(default_factory=datetime.now(UTC), nullable=False)
    message_metadata: dict = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False)
    )