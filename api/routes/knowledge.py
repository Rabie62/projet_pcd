"""
RAG Knowledge Base routes.
"""
from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from api.errors import NotFoundError, ValidationError
from api.schemas import (
    DocumentListItem,
    DocumentUploadResponse,
    KnowledgeBaseStatus,
)
from agents.graph import get_registry

router = APIRouter()


def _get_rag_system():
    """Get the RAG system instance from the agent registry."""
    registry = get_registry()
    rag_system = registry.rag_system
    if rag_system is None or not rag_system.available:
        raise HTTPException(503, "RAG knowledge base is not available")
    return rag_system


@router.get("/knowledge/status", response_model=KnowledgeBaseStatus, tags=["Knowledge Base"])
async def knowledge_status():
    """Get the status of the RAG knowledge base."""
    try:
        registry = get_registry()
        rag_system = registry.rag_system
        if rag_system is None:
            return KnowledgeBaseStatus(
                available=False, total_chunks=0,
                uploaded_documents=0, system_knowledge_indexed=False,
            )
        return KnowledgeBaseStatus(
            available=rag_system.available,
            total_chunks=rag_system.get_collection_count(),
            uploaded_documents=len(rag_system.document_registry),
            system_knowledge_indexed=rag_system.is_source_indexed("system:"),
        )
    except (OSError, RuntimeError, ConnectionError) as e:
        raise  # global handler


@router.post("/knowledge/upload", response_model=DocumentUploadResponse, tags=["Knowledge Base"])
async def upload_document(
    file: UploadFile = File(...),
    uploaded_by: str = Query(..., description="Name or ID of the uploading doctor"),
):
    """
    Upload a clinical document to the RAG knowledge base.
    Supported formats: .txt, .md, .pdf
    """
    rag = _get_rag_system()
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(400, "Uploaded file is empty")
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(400, "File too large (max 10 MB)")
    try:
        record = rag.upload_document(
            file_content=content,
            filename=file.filename or "unnamed",
            uploaded_by=uploaded_by,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return DocumentUploadResponse(
        document_id=record.document_id, filename=record.filename,
        uploaded_by=record.uploaded_by, uploaded_at=record.uploaded_at,
        file_type=record.file_type, chunk_count=record.chunk_count,
        message=f"Document '{record.filename}' uploaded and indexed ({record.chunk_count} chunks)",
    )


@router.get("/knowledge/documents", response_model=list[DocumentListItem], tags=["Knowledge Base"])
async def list_documents():
    """List all uploaded documents in the knowledge base."""
    rag = _get_rag_system()
    documents = rag.list_documents()
    return [
        DocumentListItem(
            document_id=doc.document_id, filename=doc.filename,
            uploaded_by=doc.uploaded_by, uploaded_at=doc.uploaded_at,
            file_type=doc.file_type, chunk_count=doc.chunk_count,
        )
        for doc in documents
    ]


@router.delete("/knowledge/documents/{document_id}", tags=["Knowledge Base"])
async def delete_document(document_id: str):
    """Delete an uploaded document from the knowledge base."""
    rag = _get_rag_system()
    deleted = rag.delete_document(document_id)
    if not deleted:
        raise NotFoundError(f"Document {document_id} not found")
    return {"document_id": document_id, "message": "Document deleted successfully"}
