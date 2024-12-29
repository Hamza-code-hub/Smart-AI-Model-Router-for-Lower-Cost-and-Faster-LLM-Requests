from decimal import Decimal
from functools import lru_cache

from gateway.config import load_yaml


@lru_cache
def _load_pricing() -> dict[str, dict[str, float]]:
    data = load_yaml("pricing.yaml")
    return data.get("pricing", {})


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    pricing = _load_pricing()
    entry = pricing.get(model)
    if entry is None:
        return Decimal("0")
    input_cost = Decimal(str(entry["input_per_1m_usd"])) * input_tokens / 1_000_000
    output_cost = Decimal(str(entry["output_per_1m_usd"])) * output_tokens / 1_000_000
    return input_cost + output_cost
