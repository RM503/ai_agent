from redis.asyncio import Redis

_client : Redis | None = None

def get_redis_client(
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        disable_persistence: bool = True,
        **kwargs
) -> Redis:
    global _client

    if _client is None:
        _client = Redis(host=host, port=port, db=db, decode_responses=True, **kwargs)
        if disable_persistence:
            _client.config_set("save", "")
            _client.config_set("appendonly", "no")
    return _client