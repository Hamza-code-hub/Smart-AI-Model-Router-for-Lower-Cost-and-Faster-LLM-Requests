from gateway.providers.anthropic import AnthropicProvider
from gateway.providers.base import Provider, ProviderError
from gateway.providers.ollama import OllamaProvider
from gateway.providers.openai import OpenAIProvider

_registry: dict[str, Provider] = {}


def init_providers() -> None:
    _registry["anthropic"] = AnthropicProvider()
    _registry["openai"] = OpenAIProvider()
    _registry["ollama"] = OllamaProvider()


async def close_providers() -> None:
    for provider in _registry.values():
        await provider.aclose()
    _registry.clear()


def get_provider(name: str) -> Provider:
    try:
        return _registry[name]
    except KeyError:
        raise KeyError(f"Unknown provider: {name!r}. Available: {list(_registry)}")


__all__ = ["Provider", "ProviderError", "init_providers", "close_providers", "get_provider"]
