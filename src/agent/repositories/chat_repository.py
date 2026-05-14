# Persists chat results in PostgreSQL database
from datetime import datetime, UTC
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from sqlalchemy.engine import Engine
from sqlmodel import Session, select, func

from agent.models.chat_messages import ChatMessage
from agent.models.chat_sessions import ChatSession

@dataclass(frozen=True)
class ChatRepository:
    """
    This class implements logic for persisting chat sessions and messages.

    Attributes:
        engine: SQLAlchemy engine instance
        session: SQLAlchemy session instance
        session_id: UUID
        user_id: User ID
    """
    engine: Engine
    session: Session
    session_id: UUID
    user_id: UUID | None = None

    def ensure_chat_session(self):
        """This method creates a chat session entry if it does not already exist in the table"""
        existing = self.session.get(ChatSession, self.session_id)

        if existing is None:
            # add ChatSession information
            self.session.add(
                ChatSession(
                    session_id=self.session_id,
                    user_id=self.user_id,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                    last_message_at=None,
                    session_metadata={}
                )
            )
            self.session.commit()

    def get_next_message_index(self) -> int:
        """This method generates next message index."""
        stmt = (
            select(func.max(ChatMessage.message_index))
            .where(ChatMessage.session_id == self.session_id)
        ) # important to consider each session separately
        max_index = self.session.exec(stmt).one()
        return (max_index or 0) + 1

    def insert_chat_message(
            self,
            role: Literal["human", "ai"],
            content: str,
            metadata: dict[str, Any] | None=None
        ) -> ChatMessage:
        """
        Inserts chat message at the current index

        Args:
            role (str): Current role in the conversation; must be one of 'human' or 'ai'.
            content (str): Conversation content.
            metadata (dict[str, Any]): Metadata associated with conversation.

        Returns:
            ChatMessage: A class containing important attributes pertaining to chats.
        """
        next_index = self.get_next_message_index()

        msg = ChatMessage(
            session_id=self.session_id,
            message_index=next_index,
            message_role=role,
            message_content=content,
            created_at=datetime.now(UTC),
            message_metadata=metadata
        )

        self.session.add(msg)
        session_row = self.session.get(ChatSession, self.session_id)
        if session_row is None:
            raise ValueError(f"Session {self.session_id} does not exist.")

        session_row.updated_at = datetime.now(UTC)
        session_row.last_message_at = datetime.now(UTC)
        self.session.add(session_row)

        self.session.commit()
        self.session.refresh(msg)
        return msg
