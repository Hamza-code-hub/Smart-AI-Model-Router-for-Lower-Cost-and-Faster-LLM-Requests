from pydantic import BaseModel

from gateway.config import load_yaml


class PrimaryRoute(BaseModel):
    provider: str
    model: str


class CompactionRouteConfig(BaseModel):
    enabled: bool = False
    token_threshold: int = 20000
    keep_last_n_turns: int = 4
    summarizer_virtual_model: str = "fast-qa"


class RouteConfig(BaseModel):
    virtual_model: str
    primary: PrimaryRoute
    fallback: PrimaryRoute | None = None
    compaction: CompactionRouteConfig | None = None


class RouteResolver:
    def __init__(self, routes: list[RouteConfig]) -> None:
        self._routes: dict[str, RouteConfig] = {r.virtual_model: r for r in routes}

    @classmethod
    def from_yaml(cls) -> "RouteResolver":
        data = load_yaml("routes.yaml")
        routes = [RouteConfig(**r) for r in data.get("routes", [])]
        if not routes:
            raise ValueError("routes.yaml contains no routes")
        return cls(routes)

    def resolve(self, virtual_model: str) -> RouteConfig:
        route = self._routes.get(virtual_model)
        if route is None:
            available = list(self._routes)
            raise KeyError(
                f"Virtual model {virtual_model!r} not found. Available: {available}"
            )
        return route

    def get_route(self, virtual_model: str) -> RouteConfig | None:
        return self._routes.get(virtual_model)

    def all_virtual_models(self) -> list[str]:
        return list(self._routes)

    def all_routes(self) -> list[RouteConfig]:
        return list(self._routes.values())
