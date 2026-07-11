import uuid
from datetime import datetime
from typing import Optional

import openai
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from backend.database import get_db, ChatSession, Document, User
from backend.api.auth import get_current_user
from backend.services import document_processor as dp
from backend.services.user_rag import get_service, openai_error_detail

router = APIRouter()


@router.post("/upload")
async def upload_file(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    session_id: str = Form(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # Documents are scoped to the conversation they were uploaded in.
    session = db.query(ChatSession).filter(
        ChatSession.id == session_id, ChatSession.user_id == current_user.id
    ).first()
    if not session:
        raise HTTPException(404, "Session not found")

    rag = get_service(current_user.id)

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

        if not chunks:
            raise HTTPException(422, "Could not extract text from the file")

        doc_id = chunks[0]["doc_id"]
        try:
            chunk_count = rag.add_document(chunks, summary, {
                "id": doc_id, "name": filename, "type": doc_type, "session_id": session_id,
            })
        except openai.APIError as e:
            raise HTTPException(502, openai_error_detail(e)) from e

        doc = Document(id=doc_id, user_id=current_user.id, session_id=session_id,
                       name=filename, doc_type=doc_type, chunk_count=chunk_count)
        db.add(doc)
        db.commit()

        return {"doc_id": doc_id, "name": filename, "type": doc_type, "chunk_count": chunk_count}

    elif url:
        try:
            chunks, summary = dp.process_url(url)
        except Exception as e:
            raise HTTPException(422, f"Failed to fetch URL: {e}")

        if not chunks:
            raise HTTPException(422, "No content extracted from URL")

        doc_id = chunks[0]["doc_id"]
        title  = chunks[0].get("title", url)
        try:
            chunk_count = rag.add_document(chunks, summary, {
                "id": doc_id, "name": title, "type": "url", "session_id": session_id,
            })
        except openai.APIError as e:
            raise HTTPException(502, openai_error_detail(e)) from e

        doc = Document(id=doc_id, user_id=current_user.id, session_id=session_id,
                       name=title, doc_type="url", chunk_count=chunk_count)
        db.add(doc)
        db.commit()

        return {"doc_id": doc_id, "name": title, "type": "url", "chunk_count": chunk_count}

    else:
        raise HTTPException(400, "Provide either a file or a URL")


@router.get("")
def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    docs = db.query(Document).filter(Document.user_id == current_user.id).order_by(Document.created_at.desc()).all()
    return {
        "documents": [{
            "id":          d.id,
            "name":        d.name,
            "type":        d.doc_type,
            "chunk_count": d.chunk_count,
            "created_at":  d.created_at.isoformat(),
        } for d in docs],
        "total_chunks": sum(d.chunk_count for d in docs),
    }


@router.delete("/{doc_id}")
def delete_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = db.query(Document).filter(Document.id == doc_id, Document.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    rag = get_service(current_user.id)
    rag.remove_document(doc_id)
    db.delete(doc)
    db.commit()
    return {"success": True}
