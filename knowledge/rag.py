"""
Knowledge RAG — Retrieval-Augmented Generation using Qdrant vector store.

Provides:
  - Persistent Qdrant storage (survives restarts, no external server required)
  - Document upload + indexing by doctors (PDF, TXT, Markdown)
  - Semantic retrieval for grounding LLM-generated reports
  - System knowledge auto-indexed from knowledge/ directory

Uses:
  - Qdrant (persistent local mode) as the vector store
  - pritamdeka/S-PubMedBert-MS-MARCO for medical embeddings (768d)
  - EasyOCR for scanned document support
  - PyMuPDF for high-fidelity PDF processing
"""

from __future__ import annotations

import io
import json
import os
import re
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import easyocr
import fitz  # PyMuPDF
import torch
import uuid
from loguru import logger
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchText,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    """A single retrieved knowledge chunk with relevance score."""
    text: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class DocumentRecord:
    """Metadata for an uploaded document."""
    document_id: str
    filename: str
    uploaded_by: str
    uploaded_at: str
    file_type: str
    chunk_count: int
    file_path: str


class KnowledgeRAG:
    """
    RAG system for clinical knowledge retrieval using persistent Qdrant.

    Supports two types of knowledge:
      1. System knowledge — auto-indexed from knowledge/ directory at startup
      2. Uploaded documents — added by doctors at runtime via the API

    Both types live in the same Qdrant collection and are searchable together.
    Documents persist across application restarts.
    """

    COLLECTION_NAME = "medical_knowledge"
    DEFAULT_EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
    
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

    def __init__(
        self,
        knowledge_dir: Optional[Path] = None,
        uploads_dir: Optional[Path] = None,
        qdrant_storage_dir: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        persistent: bool = True,
    ):
        self.knowledge_dir = knowledge_dir or Path(__file__).parent
        self.uploads_dir = uploads_dir or (
            self.knowledge_dir.parent / "data" / "uploads"
        )
        self.qdrant_storage_dir = qdrant_storage_dir or (
            self.knowledge_dir.parent / "data" / "qdrant_storage"
        )
        self.embedding_model_name = (
            embedding_model or self.DEFAULT_EMBEDDING_MODEL
        )
        self.persistent = persistent

        # Ensure directories exist
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        if self.persistent:
            self.qdrant_storage_dir.mkdir(parents=True, exist_ok=True)

        # Document registry (tracks uploaded documents)
        self.registry_path = self.uploads_dir / "document_registry.json"
        self.document_registry: dict[str, DocumentRecord] = {}

        self.encoder = None
        self.qdrant_client = None
        self.ocr_reader_instance = None
        self.is_available = False

        self.init_components()
        self.is_available = True
        self.load_registry()

    def init_components(self) -> None:
        """Initialise encoder and Qdrant client."""

        logger.info(
            f"Loading embedding model: {self.embedding_model_name}"
        )
        self.encoder = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = (
            self.encoder.get_sentence_embedding_dimension()
        )

        if self.persistent:
            self.qdrant_client = QdrantClient(
                path=str(self.qdrant_storage_dir)
            )
            logger.info(
                f"Qdrant client initialised (persistent: "
                f"{self.qdrant_storage_dir}), "
                f"embedding dim: {self.embedding_dim}"
            )

        # Ensure collection exists
        self.ensure_collection()

    @property
    def ocr_reader(self):
        """Lazy-loaded EasyOCR reader."""
        if self.ocr_reader_instance is None:
            logger.info("Initialising EasyOCR (English)...")
            self.ocr_reader_instance = easyocr.Reader(
                ['en'], 
                gpu=torch.cuda.is_available()
            )
        return self.ocr_reader_instance

    @property
    def available(self) -> bool:
        """Whether the RAG system is available."""
        return self.is_available

    # ── Collection management ─────────────────────────────────────────

    def ensure_collection(self) -> None:
        """Create the collection if it doesn't already exist."""

        collections = self.qdrant_client.get_collections().collections
        exists = any(
            c.name == self.COLLECTION_NAME for c in collections
        )

        if not exists:
            self.create_medical_collection()
        else:
            logger.info(f"Qdrant collection exists: {self.COLLECTION_NAME}")

    def create_medical_collection(self) -> None:
        """Create the medical knowledge collection with correct parameters."""
        self.qdrant_client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Created Qdrant collection: {self.COLLECTION_NAME} (dim: {self.embedding_dim})")

    # ── System knowledge indexing ─────────────────────────────────────

    def index_system_knowledge(self) -> int:
        """
        Index the built-in knowledge files (guidelines, tumor types,
        clinical knowledge). Only indexes if not already present.

        Returns:
            Number of new chunks indexed.
        """
        if not self.is_available:
            logger.warning("RAG not available — skipping indexing.")
            return 0

        # Check if system docs are already indexed
        if self.is_source_indexed("system:guidelines.json"):
            logger.info("System knowledge already indexed — skipping.")
            return 0

        chunks = []
        chunks.extend(self.load_json_knowledge())
        chunks.extend(self.load_markdown_knowledge())

        if not chunks:
            logger.warning("No system knowledge chunks found.")
            return 0

        # Tag all chunks as system knowledge
        for chunk in chunks:
            chunk["source"] = f"system:{chunk['source']}"
            chunk["uploaded_by"] = "system"

        count = self.index_chunks(chunks)
        logger.info(f"Indexed {count} system knowledge chunks.")
        return count

    def is_source_indexed(self, source_prefix: str) -> bool:
        """Check if documents from a given source are already indexed."""
        try:
            results = self.qdrant_client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source",
                            match=MatchText(text=source_prefix),
                        )
                    ]
                ),
                limit=1,
            )
            return len(results[0]) > 0
        except Exception:
            return False

    # ── Document upload & indexing ─────────────────────────────────────

    def upload_document(
        self,
        file_content: bytes,
        filename: str,
        uploaded_by: str,
    ) -> DocumentRecord:
        """
        Upload and index a clinical document.

        Supports: .txt, .md, .pdf

        Args:
            file_content: raw file bytes
            filename: original filename
            uploaded_by: name/ID of the uploading doctor

        Returns:
            DocumentRecord with metadata

        Raises:
            ValueError: if file type is unsupported
        """
        ext = Path(filename).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        # Generate document ID
        content_hash = hashlib.sha256(file_content).hexdigest()[:12]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        document_id = f"doc_{timestamp}_{content_hash}"

        # Save file to uploads directory
        safe_filename = f"{document_id}{ext}"
        file_path = self.uploads_dir / safe_filename
        file_path.write_bytes(file_content)

        # Parse document into text
        text_content = self.parse_document(file_content, ext)

        # Chunk the text
        chunks = self.chunk_text(
            text_content,
            source=f"upload:{filename}",
            document_id=document_id,
            uploaded_by=uploaded_by,
        )

        # Index chunks into Qdrant
        chunk_count = self.index_chunks(chunks)

        # Create and store document record
        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            uploaded_by=uploaded_by,
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            file_type=ext,
            chunk_count=chunk_count,
            file_path=str(file_path),
        )
        self.document_registry[document_id] = record
        self.save_registry()

        logger.info(
            f"Document uploaded: {filename} by {uploaded_by} "
            f"({chunk_count} chunks indexed)"
        )
        return record

    def delete_document(self, document_id: str) -> bool:
        """
        Delete an uploaded document and its indexed chunks.

        Args:
            document_id: ID of the document to delete

        Returns:
            True if deleted, False if not found
        """
        if document_id not in self.document_registry:
            return False

        record = self.document_registry[document_id]

        # Delete chunks from Qdrant
        try:
            self.qdrant_client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to delete Qdrant points: {e}")

        # Delete file from disk
        file_path = Path(record.file_path)
        if file_path.exists():
            file_path.unlink()

        # Remove from registry
        del self.document_registry[document_id]
        self.save_registry()

        logger.info(f"Document deleted: {record.filename} ({document_id})")
        return True

    def list_documents(self) -> list[DocumentRecord]:
        """List all uploaded documents."""
        return list(self.document_registry.values())

    # ── Document parsing ──────────────────────────────────────────────

    def parse_document(self, content: bytes, ext: str) -> str:
        """Parse document content based on file type."""
        if ext == ".pdf":
            # Attempt standard text extraction first
            text = self.parse_pdf(content)
            
            # If text is too short or empty, it's likely a scanned document
            if len(text.strip()) < 100:
                logger.info("PDF text extraction yielded minimal results. Falling back to OCR...")
                ocr_text = self.parse_pdf_with_ocr(content)
                if len(ocr_text) > len(text):
                    return ocr_text
            return text
        else:
            # .txt and .md — decode as UTF-8
            return content.decode("utf-8", errors="replace")

    def parse_pdf_with_ocr(self, content: bytes) -> str:
        """
        Perform OCR on PDF pages using EasyOCR.
        Converts PDF pages to images using PyMuPDF first.
        """
        if not self.ocr_reader:
            logger.warning("OCR Reader not available. Skipping OCR.")
            return ""

        text_parts = []
        # Open PDF with PyMuPDF
        doc = fitz.open(stream=content, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Render page to high-quality image (zoom=2 for better OCR)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            
            # Run EasyOCR
            results = self.ocr_reader.readtext(img_bytes, detail=0)
            page_text = " ".join(results)
            if page_text.strip():
                text_parts.append(f"--- Page {page_num + 1} ---\n{page_text}")
        
        doc.close()
        return "\n\n".join(text_parts)

    @staticmethod
    def parse_pdf(content: bytes) -> str:
        """Extract text from a PDF file."""
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
        return "\n\n".join(pages)

    # ── Text chunking ─────────────────────────────────────────────────

    def chunk_text(
        self,
        text: str,
        source: str,
        document_id: str = "",
        uploaded_by: str = "",
        max_chunk_size: int = 800,
        overlap: int = 100,
    ) -> list[dict]:
        """
        Split text into overlapping chunks for indexing.

        Uses a paragraph-aware strategy: splits on double newlines first,
        then merges small paragraphs or splits large ones.
        """
        # Split into paragraphs
        paragraphs = re.split(r"\n\s*\n", text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # If paragraph alone exceeds max, split it by sentences
            if len(paragraph) > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                sentence_chunk = ""
                for sentence in sentences:
                    if len(sentence_chunk) + len(sentence) > max_chunk_size:
                        if sentence_chunk:
                            chunks.append(sentence_chunk.strip())
                        # Overlap: keep last part of previous chunk
                        sentence_chunk = sentence_chunk[-overlap:] + " " + sentence
                    else:
                        sentence_chunk += " " + sentence
                if sentence_chunk.strip():
                    chunks.append(sentence_chunk.strip())
                continue

            # Try to merge with current chunk
            if len(current_chunk) + len(paragraph) + 2 > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Overlap: keep end of previous chunk
                current_chunk = current_chunk[-overlap:] + "\n\n" + paragraph
            else:
                current_chunk += "\n\n" + paragraph

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Filter out very short chunks
        chunks = [c for c in chunks if len(c) > 30]

        return [
            {
                "text": chunk,
                "source": source,
                "section": f"chunk_{i+1}",
                "document_id": document_id,
                "uploaded_by": uploaded_by,
            }
            for i, chunk in enumerate(chunks)
        ]

    # ── Qdrant indexing ───────────────────────────────────────────────

    def index_chunks(self, chunks: list[dict]) -> int:
        """Embed and index chunks into Qdrant."""
        if not chunks or not self.is_available:
            return 0



        texts = [c["text"] for c in chunks]
        embeddings = self.encoder.encode(texts, show_progress_bar=False)

        points = []
        for i, chunk in enumerate(chunks):
            point_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embeddings[i].tolist(),
                    payload={
                        "text": chunk["text"],
                        "source": chunk.get("source", "unknown"),
                        "section": chunk.get("section", ""),
                        "document_id": chunk.get("document_id", ""),
                        "uploaded_by": chunk.get("uploaded_by", ""),
                    },
                )
            )

        self.qdrant_client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points,
        )

        return len(points)

    # ── Retrieval ─────────────────────────────────────────────────────

    def retrieve(
        self, query: str, top_k: int = 3, min_score: float = 0.3
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant knowledge chunks for a query.

        Searches across both system knowledge and doctor-uploaded documents.
        Chunks below min_score are filtered out to avoid injecting noisy context.

        Args:
            query: Natural language query
            top_k: Maximum number of chunks to return
            min_score: Minimum cosine similarity score (0-1). Chunks below
                       this threshold are discarded.
        """
        if not self.is_available:
            return []

        count = self.get_collection_count()
        if count == 0:
            return []

        query_vector = self.encoder.encode(query).tolist()

        # Request more candidates than top_k to allow post-filtering
        candidate_k = min(top_k * 3, max(count, top_k))
        results = self.qdrant_client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=candidate_k,
        )

        # Filter by minimum score and deduplicate
        seen_texts = set()
        chunks = []
        for hit in results.points:
            if hit.score < min_score:
                continue
            text_key = hit.payload.get("text", "")[:100]
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            chunks.append(
                RetrievedChunk(
                    text=hit.payload["text"],
                    source=hit.payload.get("source", "unknown"),
                    score=hit.score,
                    metadata={
                        "section": hit.payload.get("section", ""),
                        "document_id": hit.payload.get("document_id", ""),
                        "uploaded_by": hit.payload.get("uploaded_by", ""),
                    },
                )
            )
            if len(chunks) >= top_k:
                break

        if chunks:
            scores = [c.score for c in chunks]
            logger.info(
                f"RAG retrieve '{query[:50]}...': "
                f"{len(chunks)} chunks, "
                f"scores=[{', '.join(f'{s:.3f}' for s in scores)}]"
            )

        return chunks

    def retrieve_for_findings(
        self,
        classification: Optional[str],
        tumor_detected: bool,
        confidence: float = 0.0,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant clinical knowledge based on analysis findings.

        Constructs targeted queries from the classification result and
        retrieves relevant guidelines, grading info, and treatment pathways
        from both system knowledge and doctor-uploaded documents.

        Includes RAG quality metrics in logs for traceability.
        """
        if not self.is_available:
            return []

        if self.get_collection_count() == 0:
            return []

        queries = []
        if tumor_detected and classification:
            queries.append(
                f"{classification} clinical features diagnosis treatment"
            )
            queries.append(
                f"{classification} WHO grading MRI imaging findings"
            )
            if confidence < 0.85:
                queries.append(
                    f"differential diagnosis {classification} brain tumor"
                )
        else:
            queries.append("normal brain MRI no tumor clinical correlation")

        # Retrieve and deduplicate across queries
        seen_texts = set()
        all_chunks = []
        for query in queries:
            for chunk in self.retrieve(query, top_k=3, min_score=0.3):
                text_key = chunk.text[:100]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    all_chunks.append(chunk)

        all_chunks.sort(key=lambda c: c.score, reverse=True)
        top_chunks = all_chunks[:5]

        if top_chunks:
            avg_score = sum(c.score for c in top_chunks) / len(top_chunks)
            logger.info(
                f"RAG retrieve_for_findings(classification={classification}, "
                f"tumor={tumor_detected}, conf={confidence:.2f}): "
                f"{len(top_chunks)} chunks, avg_score={avg_score:.3f}, "
                f"queries={len(queries)}"
            )

        return top_chunks

    # ── System knowledge loaders ──────────────────────────────────────

    def load_json_knowledge(self) -> list[dict]:
        """Load and chunk JSON knowledge files."""
        chunks = []

        for filename in ["guidelines.json", "tumor_types.json"]:
            filepath = self.knowledge_dir / filename
            if filepath.exists():
                with open(filepath, "r") as f:
                    data = json.load(f)
                chunks.extend(self.chunk_json(data, source=filename))

        return chunks

    def load_markdown_knowledge(self) -> list[dict]:
        """Load and chunk Markdown knowledge files."""
        chunks = []

        md_path = self.knowledge_dir / "clinical_knowledge.md"
        if md_path.exists():
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            chunks.extend(
                self.chunk_markdown(content, source="clinical_knowledge.md")
            )

        return chunks

    @staticmethod
    def chunk_json(
        data: dict, source: str, prefix: str = ""
    ) -> list[dict]:
        """Recursively chunk a JSON object into text passages."""
        chunks = []

        for key, value in data.items():
            path = f"{prefix}/{key}" if prefix else key

            if isinstance(value, dict):
                flat_text = KnowledgeRAG.flatten_dict(value, key)
                if len(flat_text) > 50:
                    chunks.append({
                        "text": flat_text,
                        "source": source,
                        "section": path,
                    })
                sub_chunks = KnowledgeRAG.chunk_json(
                    value, source, prefix=path
                )
                chunks.extend(sub_chunks)

            elif isinstance(value, list):
                list_text = f"{key}: {', '.join(str(v) for v in value)}"
                if len(list_text) > 30:
                    chunks.append({
                        "text": list_text,
                        "source": source,
                        "section": path,
                    })

            elif isinstance(value, str) and len(value) > 50:
                chunks.append({
                    "text": f"{key}: {value}",
                    "source": source,
                    "section": path,
                })

        return chunks

    @staticmethod
    def flatten_dict(d: dict, title: str = "") -> str:
        """Flatten a dict into a readable text passage."""
        parts = []
        if title:
            parts.append(f"{title}:")
        for k, v in d.items():
            if isinstance(v, dict):
                sub = ", ".join(f"{sk}: {sv}" for sk, sv in v.items())
                parts.append(f"  {k}: {sub}")
            elif isinstance(v, list):
                parts.append(f"  {k}: {', '.join(str(i) for i in v)}")
            else:
                parts.append(f"  {k}: {v}")
        return "\n".join(parts)

    @staticmethod
    def chunk_markdown(content: str, source: str) -> list[dict]:
        """Split Markdown into section-based chunks."""
        sections = re.split(r"\n##\s+", content)
        chunks = []

        for section in sections:
            section = section.strip()
            if not section or len(section) < 50:
                continue

            lines = section.split("\n", 1)
            title = lines[0].strip().lstrip("#").strip()
            body = lines[1].strip() if len(lines) > 1 else section

            body = re.sub(r"\*\*(.+?)\*\*", r"\1", body)
            body = re.sub(r"\*(.+?)\*", r"\1", body)
            body = body.replace("---", "").strip()

            if len(body) > 50:
                chunks.append({
                    "text": f"{title}\n{body}",
                    "source": source,
                    "section": title,
                })

        return chunks

    # ── Document registry persistence ─────────────────────────────────

    def save_registry(self) -> None:
        """Save document registry to disk."""
        data = {}
        for doc_id, record in self.document_registry.items():
            data[doc_id] = {
                "document_id": record.document_id,
                "filename": record.filename,
                "uploaded_by": record.uploaded_by,
                "uploaded_at": record.uploaded_at,
                "file_type": record.file_type,
                "chunk_count": record.chunk_count,
                "file_path": record.file_path,
            }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_registry(self) -> None:
        """Load document registry from disk."""
        if not self.registry_path.exists():
            return
        with open(self.registry_path, "r") as f:
            data = json.load(f)
        for doc_id, info in data.items():
            self.document_registry[doc_id] = DocumentRecord(**info)
        logger.info(
            f"Loaded {len(self.document_registry)} document records."
        )

    # ── Utilities ─────────────────────────────────────────────────────

    def get_collection_count(self) -> int:
        """Get the number of points in the collection."""
        try:
            info = self.qdrant_client.get_collection(self.COLLECTION_NAME)
            return info.points_count
        except Exception:
            return 0

    def format_context(
        self, chunks: list[RetrievedChunk], include_scores: bool = True
    ) -> str:
        """Format retrieved chunks into a context string for LLM prompts.

        Args:
            chunks: Retrieved knowledge chunks
            include_scores: Whether to include relevance scores (for traceability)
        """
        if not chunks:
            return ""

        parts = ["RELEVANT CLINICAL KNOWLEDGE:"]
        for i, chunk in enumerate(chunks, 1):
            source_label = chunk.source
            if chunk.metadata.get("uploaded_by"):
                source_label += (
                    f", uploaded by {chunk.metadata['uploaded_by']}"
                )
            score_info = f", relevance: {chunk.score:.2f}" if include_scores else ""
            parts.append(
                f"\n[{i}] (source: {source_label}{score_info})\n{chunk.text}"
            )

        return "\n".join(parts)
