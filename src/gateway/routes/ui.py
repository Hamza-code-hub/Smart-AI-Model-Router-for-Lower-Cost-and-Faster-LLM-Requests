from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from gateway.routing import get_resolver

router = APIRouter()

_STATIC_DIR = Path(__file__).parent.parent / "static"

_MODEL_META: dict[str, dict[str, str]] = {
    "claude-haiku-4-5":  {"display_name": "Claude Haiku 4.5",  "description": "Fast · Cheap"},
    "claude-sonnet-4-6": {"display_name": "Claude Sonnet 4.6", "description": "Balanced"},
    "claude-opus-4-8":   {"display_name": "Claude Opus 4.8",   "description": "Most capable"},
    "claude-opus-4-7":   {"display_name": "Claude Opus 4.7",   "description": "Opus · Prev gen"},
    "claude-opus-4-6":   {"display_name": "Claude Opus 4.6",   "description": "Opus · Older"},
    "claude-opus-4-5":   {"display_name": "Claude Opus 4.5",   "description": "Opus · Legacy"},
    "claude-fable-5":    {"display_name": "Claude Fable 5",     "description": "Latest"},
    "gpt-4o":            {"display_name": "GPT-4o",             "description": "Fast · Vision"},
    "gpt-4o-mini":       {"display_name": "GPT-4o Mini",        "description": "Cheapest"},
    "o1":                {"display_name": "o1",                  "description": "Reasoning"},
    "o3":                {"display_name": "o3",                  "description": "Latest"},
}


@router.get("/chat", include_in_schema=False)
async def serve_chat_ui() -> FileResponse:
    # no-cache forces the browser to revalidate; the page embeds its JS inline,
    # so a stale HTML copy means stale JS. ETag still yields cheap 304s.
    return FileResponse(_STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@router.get("/ui/models", include_in_schema=False)
async def list_ui_models() -> JSONResponse:
    resolver = get_resolver()
    models = [
        {
            "id": "auto",
            "provider": "auto",
            "model": "auto",
            "display_name": "Auto",
            "description": "Smart routing · picks the right model",
        }
    ]
    for route in resolver.all_routes():
        vm = route.virtual_model
        meta = _MODEL_META.get(route.primary.model, {})
        if vm != route.primary.model:
            continue
        models.append({
            "id": vm,
            "provider": route.primary.provider,
            "model": route.primary.model,
            "display_name": meta.get("display_name", vm),
            "description": meta.get("description", ""),
        })
    return JSONResponse({"models": models})
