from fastapi import APIRouter

from gateway.models import ModelCard, ModelList
from gateway.routing import get_resolver

router = APIRouter()


@router.get("/models", response_model=ModelList)
async def list_models() -> ModelList:
    resolver = get_resolver()
    cards = [ModelCard(id=vm) for vm in resolver.all_virtual_models()]
    return ModelList(data=cards)
