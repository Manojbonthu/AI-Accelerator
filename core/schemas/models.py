"""
core/schemas/models.py

Defines all data structures for the PDF extraction pipeline.
Everything downstream (handlers, chunker, storage) uses these schemas.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid


class BlockType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    IMAGE = "image"
    CAPTION = "caption"
    PAGE_BREAK = "page_break"


class ChunkType(str, Enum):
    PARAGRAPH = "paragraph"
    TABLE = "table"
    IMAGE_DESCRIPTION = "image_description"
    MIXED = "mixed"


@dataclass
class TableBlock:
    headers: List[str]
    rows: List[List[str]]
    caption: Optional[str] = None
    page: int = 1
    note: Optional[str] = None


@dataclass
class ImageBlock:
    image_bytes: Optional[bytes] = None
    image_path: Optional[str] = None
    mime_type: str = "image/png"
    caption: Optional[str] = None
    description: Optional[str] = None
    page: int = 1
    confidence: float = 1.0


@dataclass
class ContentBlock:
    type: BlockType
    text: str = ""
    level: Optional[int] = None
    page: int = 1
    section_id: Optional[str] = None
    table: Optional[TableBlock] = None
    image: Optional[ImageBlock] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Section:
    title: str
    level: int
    section_path: List[str]
    page_start: int
    page_end: int
    content: str = ""
    tables: List[TableBlock] = field(default_factory=list)
    figures: List[ImageBlock] = field(default_factory=list)
    children: List['Section'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "level": self.level,
            "section_path": self.section_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "content": self.content,
            "tables": [t.__dict__ for t in self.tables],
            "figures": [f.__dict__ for f in self.figures],
            "children": [c.to_dict() for c in self.children]
        }


@dataclass
class NormalizedDocument:
    file_name: str
    total_pages: int
    pdf_type: str
    blocks: List[ContentBlock] = field(default_factory=list)
    sections: List[Section] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    chunk_index: int
    document_id: str
    document_name: str
    document_type: str
    domain: str = "general"
    section: str = "(No Heading)"
    subsection: str = ""
    section_level: int = 1
    chunk_type: str = "paragraph"
    chunk_title: str = "(No Heading)"
    content: str = ""
    embedding_text: str = ""
    page_start: int = 1
    page_end: int = 1
    tables: List[Dict[str, Any]] = field(default_factory=list)
    figures: List[Dict[str, Any]] = field(default_factory=list)
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    relationships: Dict[str, Any] = field(default_factory=lambda: {
        "parent_chunk_id": None,
        "previous_chunk_id": None,
        "next_chunk_id": None
    })

    def to_dict(self) -> Dict[str, Any]:
        def clean(obj):
            if isinstance(obj, bytes):
                return None
            elif isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(clean(v) for v in obj)
            else:
                return obj

        return clean({
            "document_id": self.document_id,
            "document_name": self.document_name,
            "document_type": self.document_type,
            "domain": self.domain,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "section": self.section,
            "subsection": self.subsection,
            "section_level": self.section_level,
            "chunk_type": self.chunk_type,
            "chunk_title": self.chunk_title,
            "content": self.content,
            "embedding_text": self.embedding_text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "tables": self.tables,
            "figures": self.figures,
            "token_count": self.token_count,
            "metadata": self.metadata,
            "relationships": self.relationships
        })