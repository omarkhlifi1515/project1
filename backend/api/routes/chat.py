from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.api.deps.auth import get_current_user
from backend.db.base import get_db
from backend.db.models import Message, User
from backend.schemas.chat import ChatRequest, ChatResponse
from backend.services.rag_service import rag_service
from backend.config.settings import get_settings

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/query", response_model=ChatResponse)
def query_chat(
    payload: ChatRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    settings = get_settings()
    model_preset = payload.model_preset or settings.default_model_preset
    answer, sources = rag_service.answer(payload.query, model_preset=model_preset)
    db.add(Message(user_id=user.id, role="user", content=payload.query))
    db.add(Message(user_id=user.id, role="assistant", content=answer))
    db.commit()
    return ChatResponse(answer=answer, sources=sources)


@router.post("/stream")
def stream_chat(
    payload: ChatRequest,
    user: User = Depends(get_current_user),
):
    settings = get_settings()
    model_preset = payload.model_preset or settings.default_model_preset

    def event_stream():
        for token, sources in rag_service.stream(payload.query, model_preset=model_preset):
            if token is not None:
                yield f"data: {token}\n\n"
            if sources is not None:
                yield f"event: sources\ndata: {' | '.join(sources)}\n\n"
        yield "event: done\ndata: done\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
