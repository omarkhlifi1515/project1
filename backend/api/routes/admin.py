from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.api.deps.auth import require_role
from backend.config.settings import get_settings
from backend.db.base import get_db
from backend.db.models import AuditLog, User
from backend.schemas.admin import SyncResponse
from backend.jobs.manager import job_manager
from backend.services.rag_service import rag_service

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/sync-index", response_model=SyncResponse)
def sync_index(
    admin_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    settings = get_settings()
    rag_service.sync_index(settings.default_model_preset)
    db.add(
        AuditLog(
            actor_email=admin_user.email,
            action="sync_index",
            details="Manual index sync triggered from admin endpoint.",
        )
    )
    db.commit()
    return SyncResponse(status="ok", detail="Index sync completed")


@router.post("/jobs/refresh")
def run_refresh_job(admin_user: User = Depends(require_role("admin")), db: Session = Depends(get_db)):
    result = job_manager.run_refresh_pipeline()
    db.add(
        AuditLog(
            actor_email=admin_user.email,
            action="run_refresh_job",
            status="success" if result.get("accepted") else "rejected",
            details=result.get("detail", ""),
        )
    )
    db.commit()
    return result


@router.get("/jobs/status")
def refresh_job_status(admin_user: User = Depends(require_role("admin"))):
    return job_manager.status()
