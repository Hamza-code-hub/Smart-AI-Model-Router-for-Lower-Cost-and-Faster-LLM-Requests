import redis.asyncio as aioredis

from gateway.config import get_settings

_redis: aioredis.Redis | None = None  # type: ignore[type-arg]


async def init_redis() -> None:
    global _redis
    settings = get_settings()
    _redis = aioredis.from_url(settings.redis_url, decode_responses=True)


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


def get_redis() -> aioredis.Redis:  # type: ignore[type-arg]
    if _redis is None:
        raise RuntimeError("Redis client not initialized")
    return _redis
