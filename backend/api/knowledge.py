"""
Global knowledge base management.

Documents added here are shared across ALL users and sessions — every chat
query retrieves from them in addition to the session's own attachments.
For bulk loading, prefer scripts/ingest_knowledge.py.
"""

import os
from typing import Optional

import openai
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from backend.database import User
from backend.api.auth import get_current_user
from backend.services import document_processor as dp
from backend.services.global_kb import get_kb
from backend.services.user_rag import openai_error_detail

router = APIRouter()


def require_kb_admin(current_user: User = Depends(get_current_user)) -> User:
    """If ADMIN_EMAILS (comma-separated) is set in the environment, only
    those accounts may modify the shared KB; otherwise any signed-in user
    can — convenient for development, lock it down in production."""
    admins = {e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()}
    if admins and current_user.email.lower() not in admins:
        raise HTTPException(403, "Chỉ quản trị viên mới được thay đổi kho kiến thức chung")
    return current_user


@router.post("/upload")
async def upload_knowledge(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    current_user: User = Depends(require_kb_admin),
):
    kb = get_kb()

    if file:
        filename = file.filename or "upload"
        ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in dp.ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type: {ext}")

        file_bytes = await file.read()
        try:
            chunks, summary, doc_type = dp.process_file(file_bytes, filename, file.content_type or "")
        except openai.APIError as e:
            raise HTTPException(502, openai_error_detail(e)) from e
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        name = filename

    elif url:
        try:
            chunks, summary = dp.process_url(url)
        except Exception as e:
            raise HTTPException(422, f"Failed to fetch URL: {e}") from e
        doc_type = "url"
        name = chunks[0].get("title", url) if chunks else url

    else:
        raise HTTPException(400, "Provide either a file or a URL")

    if not chunks:
        raise HTTPException(422, "Could not extract text from the source")

    doc_id = chunks[0]["doc_id"]
    try:
        chunk_count = kb.add_document(chunks, summary, {"id": doc_id, "name": name, "type": doc_type})
    except openai.APIError as e:
        raise HTTPException(502, openai_error_detail(e)) from e

    return {"doc_id": doc_id, "name": name, "type": doc_type,
            "chunk_count": chunk_count, "scope": "global"}


@router.get("")
def list_knowledge(current_user: User = Depends(get_current_user)):
    kb = get_kb()
    return {
        "documents": [{
            "id":          d["id"],
            "name":        d["name"],
            "type":        d["type"],
            "chunk_count": d["chunk_count"],
            "created_at":  d.get("created_at", ""),
        } for d in kb.get_documents()],
        "total_chunks": kb.chunk_count(),
    }


@router.delete("/{doc_id}")
def delete_knowledge(
    doc_id: str,
    current_user: User = Depends(require_kb_admin),
):
    kb = get_kb()
    if not any(d["id"] == doc_id for d in kb.get_documents()):
        raise HTTPException(404, "Document not found")
    kb.remove_document(doc_id)
    return {"success": True}
