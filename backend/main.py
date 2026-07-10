from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import create_tables
from backend.api.auth import router as auth_router
from backend.api.documents import router as docs_router
from backend.api.chat import router as chat_router

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


@app.get("/health")
async def health():
    return {"status": "ok"}


app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(docs_router, prefix="/api/documents", tags=["documents"])
app.include_router(chat_router, prefix="/api", tags=["chat"])
