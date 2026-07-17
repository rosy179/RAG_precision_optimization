from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.database import create_tables
from backend.api.auth import router as auth_router
from backend.api.documents import router as docs_router
from backend.api.chat import router as chat_router
from backend.api.knowledge import router as knowledge_router
from backend.api.monitoring import router as monitoring_router

app = FastAPI(title="IT RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    create_tables()
    from backend.services.user_rag import warm_up
    warm_up()
    # Restore the shared knowledge base into memory at boot (BM25 index,
    # doc registry) instead of on the first chat request.
    from backend.services.global_kb import get_kb
    get_kb()


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(docs_router, prefix="/api/documents", tags=["documents"])
app.include_router(knowledge_router, prefix="/api/knowledge", tags=["knowledge"])
app.include_router(monitoring_router, prefix="/api/monitoring", tags=["monitoring"])
app.include_router(chat_router, prefix="/api", tags=["chat"])

# ── Serve the built frontend (single-origin deploys: Cloudflare Tunnel, VPS) ──
# API routes above win; anything else falls through to the SPA. In dev the
# Vite server (5173) is used instead and this block is inert if dist/ is absent.
FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"

if FRONTEND_DIST.exists():
    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        candidate = (FRONTEND_DIST / full_path).resolve()
        # Path-traversal guard: only serve files inside dist/
        if candidate.is_file() and candidate.is_relative_to(FRONTEND_DIST.resolve()):
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
