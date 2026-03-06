# Redis history manager for session conversations
import uuid

from langchain_community.chat_message_histories import RedisChatMessageHistory

class RedisHistoryManager:
    """Custom class for storing session information on redis."""
    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url

    def __call__(self, session_id: str | uuid.UUID) -> RedisChatMessageHistory:
        """
        This function creates a session information on redis when
        `RedisChatMessageHistory` is called. This can then be used in
        `RunnableWithMessageHistory`.

        Args:
            session_id (str | uuid.UUID): session id
        Returns:
            RedisChatMessageHistory: session information
        """
        if isinstance(session_id, uuid.UUID):
            # convert session_id into str if passed as uuid object
            session_id = str(session_id)
        return RedisChatMessageHistory(
            session_id=session_id,
            url=self.redis_url
        )

    def clear(self, session_id: str | uuid.UUID) -> None:
        """Clears session information on redis for given session_id."""
        if isinstance(session_id, uuid.UUID):
            # convert session_id into str if passed as uuid object
            session_id = str(session_id)
        history = RedisChatMessageHistory(
            session_id=session_id,
            url=self.redis_url
        )

        history.clear()