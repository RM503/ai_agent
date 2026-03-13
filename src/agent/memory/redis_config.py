from redis.asyncio import Redis

def get_redis_client(db: int=0, **kwargs) -> Redis:
    return Redis(
        host="localhost",
        port=6379,
        db=db,
        decode_responses=True,
        **kwargs
    )