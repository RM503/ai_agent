# Redis configuration

from __future__ import annotations

from redis.asyncio import Redis

_clients: dict[str, Redis] = {}

def get_redis_client(db: int=0, **kwargs) -> Redis:
    """Get a Redis client for the given name."""
    if db not in _clients:
        _clients[db] = Redis(
            host="localhost",
            port=6379,
            db=0,
            decode_responses=True,
            **kwargs
        )
    return _clients[db]

# Redis clients for different purposes
redis_main = get_redis_client(db=0)
redis_jobs = get_redis_client(db=1)
redis_cache = get_redis_client(db=2)