from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=2)
    model_preset: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
