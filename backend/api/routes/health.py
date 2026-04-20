from fastapi import APIRouter

from backend.config.settings import get_settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health():
    settings = get_settings()
    return {
        "status": "ok",
        "environment": settings.environment,
        "default_model_preset": settings.default_model_preset,
    }
