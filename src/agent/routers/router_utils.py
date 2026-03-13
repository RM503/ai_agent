import json

from redis import RedisError
from redis.asyncio import Redis 

from agent.common.logging_config import get_logger

logger = get_logger(__name__)

r = Redis(host="localhost", port=6379, db=3, decode_responses=True)

async def retrieve_uploaded_artifacts(session_id: str) -> dict[str, str]:
    """Retrieve uploaded artifacts using session_id"""
    try:
        key = f"session_id:{session_id}:file"
        payload = await r.get(key)

        return payload
    except RedisError as e:
        logger.exception(f"Failed to retrieve uploaded artifacts: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.exception(f"Failed to retrieve uploaded artifacts: {e}")
        return None