"""
core/schemas/models.py

Defines all data structures for the PDF extraction pipeline.
Everything downstream (handlers, chunker, storage) uses these schemas.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid


# ──────────────────────────────────────────────
# 1. ENUMS
# ──────────────────────────────────────────────

class BlockType(str, Enum):
    """Types of content that can appear in a document."""
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    IMAGE = "image"
    CAPTION = "caption"
    PAGE_BREAK = "page_break"


class ChunkType(str, Enum):
    """Types of chunks produced after chunking."""
    PARAGRAPH = "paragraph"
    TABLE = "table"
    IMAGE_DESCRIPTION = "image_description"
    MIXED = "mixed"


# ──────────────────────────────────────────────
# 2. CONTENT BLOCKS (from extraction)
# ──────────────────────────────────────────────

@dataclass
class TableBlock:
    """Structured table extracted from a PDF."""
    headers: List[str]
    rows: List[List[str]]
    caption: Optional[str] = None
    page: int = 1
    note: Optional[str] = None


@dataclass
class ImageBlock:
    """Image extracted from a PDF page."""
    image_bytes: Optional[bytes] = None
    image_path: Optional[str] = None
    mime_type: str = "image/png"
    caption: Optional[str] = None
    description: Optional[str] = None  # Gemma-generated description
    page: int = 1
    confidence: float = 1.0  # 1.0 for digital, OCR confidence for scanned


@dataclass
class ContentBlock:
    """
    A single piece of content from a PDF page.
    Can be text, heading, table, image, etc.
    """
    type: BlockType
    text: str = ""
    level: Optional[int] = None        # Heading level (1, 2, 3...)
    page: int = 1
    section_id: Optional[str] = None   # Links to parent heading
    table: Optional[TableBlock] = None
    image: Optional[ImageBlock] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────
# 3. NORMALIZED DOCUMENT (output of handlers)
# ──────────────────────────────────────────────

@dataclass
class NormalizedDocument:
    """
    The complete, structured output after extraction.
    Same shape regardless of PDF type (digital/scanned/mixed).
    """
    file_name: str
    total_pages: int
    pdf_type: str                     # "digital", "scanned", or "mixed"
    blocks: List[ContentBlock] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────
# 4. CHUNK (output of chunker, ready for Qdrant)
# ──────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A chunk ready for embedding and Qdrant storage.
    Self-contained with all context needed for accurate retrieval.
    """
    chunk_id: str
    chunk_index: int
    document_name: str
    section_path: str                 # "Section > Subsection"
    chunk_type: str                   # "paragraph", "table", "image_description", "mixed"
    content: str                      # Raw content (text or markdown table)
    embedding_text: str               # Content + context headers (THIS gets embedded)
    page_start: int
    page_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for JSON/Qdrant payload."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "document_name": self.document_name,
            "section_path": self.section_path,
            "chunk_type": self.chunk_type,
            "content": self.content,
            "embedding_text": self.embedding_text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "metadata": self.metadata
        }


# ──────────────────────────────────────────────
# 5. HELPER FUNCTIONS
# ──────────────────────────────────────────────

def create_chunk(
    chunk_index: int,
    document_name: str,
    section_path: str,
    chunk_type: str,
    content: str,
    page_start: int,
    page_end: int,
    metadata: Optional[Dict[str, Any]] = None
) -> Chunk:
    """
    Factory function to create a Chunk with auto-generated ID
    and properly formatted embedding_text.
    """
    chunk_id = str(uuid.uuid4())
    
    # Build embedding text with context headers
    embedding_text = f"""Document: {document_name}
Section: {section_path}
Pages: {page_start}-{page_end}

{content}"""
    
    return Chunk(
        chunk_id=chunk_id,
        chunk_index=chunk_index,
        document_name=document_name,
        section_path=section_path,
        chunk_type=chunk_type,
        content=content,
        embedding_text=embedding_text,
        page_start=page_start,
        page_end=page_end,
        metadata=metadata or {}
    )