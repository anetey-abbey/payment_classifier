from fastapi import APIRouter

from app.api.routes import classification

api_router = APIRouter()

api_router.include_router(classification.router)


def get_api_router() -> APIRouter:
    return api_router
