from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.core.database import Base, SessionLocal, engine
from app.core.settings import settings
from app.db.schemas import ChatRequest, ChatResponse, MemoryCreateRequest, MemoryView
from app.services.llm_client import LLMUnavailableError, OllamaClient
from app.services.memory_service import MemoryService


app = FastAPI(title=settings.app_title)
web_dir = Path(__file__).resolve().parent / "web"
templates = Jinja2Templates(directory=str(web_dir / "templates"))
app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")

Base.metadata.create_all(bind=engine)
memory_service = MemoryService(OllamaClient(settings.ollama_model, settings.ollama_host))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "title": settings.app_title},
    )


@app.get("/api/health")
def health():
    try:
        memory_service.llm_client.health_check()
        return {"status": "ok", "llm": "ready", "model": settings.ollama_model}
    except LLMUnavailableError as exc:
        return {
            "status": "degraded",
            "llm": "unavailable",
            "model": settings.ollama_model,
            "detail": str(exc),
        }


@app.get("/api/memories", response_model=list[MemoryView])
def list_memories(db: Session = Depends(get_db)):
    return memory_service.list_memories(db)


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    try:
        reply, used, created = memory_service.handle_chat_turn(db, payload.message)
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive HTTP wrapper
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ChatResponse(reply=reply, memories_used=used, new_memories=created)


@app.post("/api/memories", response_model=MemoryView)
def create_memory(payload: MemoryCreateRequest, db: Session = Depends(get_db)):
    return memory_service.add_manual_memory(db, payload.content, payload.kind)


@app.post("/api/memories/remember-latest", response_model=MemoryView)
def remember_latest(db: Session = Depends(get_db)):
    memory = memory_service.store_latest_user_message(db)
    if memory is None:
        raise HTTPException(status_code=404, detail="No user message found to remember.")
    return memory


@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: int, db: Session = Depends(get_db)):
    deleted = memory_service.delete_memory(db, memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found.")
    return {"status": "deleted"}
