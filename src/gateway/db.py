from contextlib import asynccontextmanager
from typing import AsyncGenerator

import asyncpg

from gateway.config import get_settings

_pool: asyncpg.Pool | None = None


async def init_db() -> None:
    global _pool
    settings = get_settings()
    # asyncpg uses postgresql:// not postgresql+asyncpg://
    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    _pool = await asyncpg.create_pool(dsn, min_size=2, max_size=settings.database_pool_size)


async def close_db() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool not initialized")
    return _pool


@asynccontextmanager
async def acquire() -> AsyncGenerator[asyncpg.Connection, None]:
    pool = get_pool()
    async with pool.acquire() as conn:
        yield conn
