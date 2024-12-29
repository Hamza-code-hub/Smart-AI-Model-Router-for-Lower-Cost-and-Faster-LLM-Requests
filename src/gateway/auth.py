import hashlib
import uuid
from dataclasses import dataclass
from functools import lru_cache

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from gateway.config import get_settings, load_yaml

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class TenantInfo:
    tenant_id: str
    api_key_id: uuid.UUID


@dataclass(frozen=True)
class TenantLimits:
    rate_limit_rpm: int = 0  # 0 = unlimited
    daily_token_budget: int | None = None
    monthly_cost_cap_usd: float | None = None


@lru_cache
def _load_key_index() -> dict[str, TenantInfo]:
    """Build a hash → TenantInfo lookup from tenants.yaml. Cached at startup."""
    data = load_yaml("tenants.yaml")
    index: dict[str, TenantInfo] = {}
    for tenant in data.get("tenants", []):
        for key_entry in tenant.get("api_keys", []):
            key_hash = key_entry.get("key_hash", "")
            if key_hash:
                index[key_hash] = TenantInfo(
                    tenant_id=tenant["id"],
                    api_key_id=uuid.UUID(int=abs(hash(key_entry["key_id"])) % (2**128)),
                )
    return index


@lru_cache
def _load_limits_index() -> dict[str, TenantLimits]:
    """Build tenant_id → TenantLimits from tenants.yaml."""
    data = load_yaml("tenants.yaml")
    index: dict[str, TenantLimits] = {}
    for tenant in data.get("tenants", []):
        index[tenant["id"]] = TenantLimits(
            rate_limit_rpm=int(tenant.get("rate_limit_rpm", 0) or 0),
            daily_token_budget=(
                int(tenant["daily_token_budget"])
                if tenant.get("daily_token_budget") is not None
                else None
            ),
            monthly_cost_cap_usd=(
                float(tenant["monthly_cost_cap_usd"])
                if tenant.get("monthly_cost_cap_usd") is not None
                else None
            ),
        )
    return index


def get_tenant_limits(tenant_id: str) -> TenantLimits | None:
    return _load_limits_index().get(tenant_id)


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


async def require_tenant(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> TenantInfo:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    key_hash = _hash_key(credentials.credentials)
    index = _load_key_index()
    tenant = index.get(key_hash)

    if tenant is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return tenant


async def require_admin(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    settings = get_settings()
    if credentials.credentials != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
