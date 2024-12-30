from gateway.routing.resolver import RouteResolver

_resolver: RouteResolver | None = None


def init_resolver() -> None:
    global _resolver
    _resolver = RouteResolver.from_yaml()


def get_resolver() -> RouteResolver:
    if _resolver is None:
        raise RuntimeError("RouteResolver not initialized")
    return _resolver
