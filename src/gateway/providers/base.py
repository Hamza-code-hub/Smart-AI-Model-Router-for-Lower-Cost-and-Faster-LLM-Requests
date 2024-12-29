from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from gateway.models import ChatCompletionRequest, ChatCompletionResponse


class ProviderError(Exception):
    def __init__(self, status_code: int, message: str, provider: str) -> None:
        self.status_code = status_code
        self.message = message
        self.provider = provider
        super().__init__(f"[{provider}] HTTP {status_code}: {message}")


class Provider(ABC):
    name: str

    @abstractmethod
    async def complete(self, request: ChatCompletionRequest, model: str) -> ChatCompletionResponse:
        ...

    @abstractmethod
    def stream(self, request: ChatCompletionRequest, model: str) -> AsyncIterator[str]:
        # Implementations are async generators that yield SSE lines: "data: {...}\n\n"
        ...

    async def aclose(self) -> None:
        pass
