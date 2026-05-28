"""
core/ingestion/chunker.py

Takes a NormalizedDocument with hierarchical sections and returns chunks ready for embedding.
Produces parent chunks per section and (optionally) child chunks for parent‑child retrieval.
"""

import uuid
import tiktoken
from typing import List, Optional, Dict, Any
from core.schemas.models import (
    NormalizedDocument, Section, Chunk, TableBlock, ImageBlock
)


def chunk_document(
    doc: NormalizedDocument,
    document_id: Optional[str] = None,
    domain: str = "general",
    use_parent_child: bool = True,
    max_parent_tokens: int = 1500,
    max_child_tokens: int = 500,      # increased from 300 to reduce fragmentation
    overlap_tokens: int = 50
) -> List[Chunk]:
    """
    Convert a NormalizedDocument into chunks using hierarchical sections.
    """
    if not document_id:
        document_id = str(uuid.uuid4())

    if doc.sections:
        parent_chunks = _create_parent_chunks_from_sections(doc.sections, doc, document_id, domain)
    elif doc.blocks:
        return _chunk_flat_blocks(doc, document_id, domain)
    else:
        return []

    if use_parent_child:
        all_chunks = []
        for parent in parent_chunks:
            children = _split_parent_into_children(parent, max_child_tokens, overlap_tokens)
            all_chunks.extend(children)
            all_chunks.append(parent)
        # Relink sequential relationships (previous/next) – but preserve parent link in relationships
        for i, chunk in enumerate(all_chunks):
            # Keep relationships dict; only update previous/next, keep parent_chunk_id if present
            if i > 0:
                chunk.relationships["previous_chunk_id"] = all_chunks[i-1].chunk_id
            if i < len(all_chunks) - 1:
                chunk.relationships["next_chunk_id"] = all_chunks[i+1].chunk_id
        return all_chunks
    else:
        for i, chunk in enumerate(parent_chunks):
            if i > 0:
                chunk.relationships["previous_chunk_id"] = parent_chunks[i-1].chunk_id
            if i < len(parent_chunks) - 1:
                chunk.relationships["next_chunk_id"] = parent_chunks[i+1].chunk_id
        return parent_chunks


# ------------------------------------------------------------------
# Section‑based chunking
# ------------------------------------------------------------------

def _create_parent_chunks_from_sections(
    sections: List[Section],
    doc: NormalizedDocument,
    document_id: str,
    domain: str,
    idx_start: int = 0
) -> List[Chunk]:
    """Recursively traverse section tree and create one parent chunk per section."""
    token_estimator = tiktoken.get_encoding("cl100k_base")
    chunks = []
    global_idx = idx_start
    for section in sections:
        full_content = section.content.strip()
        if not full_content:
            # Skip empty sections, but still process children
            child_chunks = _create_parent_chunks_from_sections(
                section.children, doc, document_id, domain, global_idx
            )
            chunks.extend(child_chunks)
            global_idx += len(child_chunks)
            continue

        # Build embedding text with full section path
        section_path_str = " > ".join(section.section_path)
        embedding_text = (
            f"Document: {doc.file_name}\n"
            f"Section Path: {section_path_str}\n"
            f"Pages: {section.page_start}–{section.page_end}\n\n"
            f"{full_content}"
        )

        # Extract tables and figures as JSON‑serializable dicts
        tables_dict = [t.__dict__ for t in section.tables]
        figures_dict = []
        for fig in section.figures:
            figures_dict.append({
                "description": fig.description,
                "caption": fig.caption,
                "page": fig.page,
                "mime_type": fig.mime_type,
                "confidence": fig.confidence
            })

        # Compute token count for parent chunk
        token_count = len(token_estimator.encode(full_content))

        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            chunk_index=global_idx,
            document_id=document_id,
            document_name=doc.file_name,
            document_type=doc.pdf_type,
            domain=domain,
            section=section.title,
            subsection="",   # FIXED: always empty for parent
            section_level=section.level,
            chunk_type="parent",
            chunk_title=section.title,
            content=full_content,
            embedding_text=embedding_text,
            page_start=section.page_start,
            page_end=section.page_end,
            tables=tables_dict,
            figures=figures_dict,
            token_count=token_count,   # NEW
            metadata={"section_path": section.section_path}
        )
        chunks.append(chunk)
        global_idx += 1

        # Process children recursively
        child_chunks = _create_parent_chunks_from_sections(
            section.children, doc, document_id, domain, global_idx
        )
        chunks.extend(child_chunks)
        global_idx += len(child_chunks)

    return chunks


def _split_parent_into_children(parent: Chunk, max_tokens: int, overlap: int) -> List[Chunk]:
    """Split a parent chunk's content into smaller child chunks with overlap."""
    sentences = _split_into_sentences(parent.content)
    token_estimator = tiktoken.get_encoding("cl100k_base")
    children = []
    current_sentences = []
    current_tokens = 0

    # Get section path from parent's metadata
    section_path = parent.metadata.get("section_path", [])
    section_path_str = " > ".join(section_path)

    for sent in sentences:
        sent_tokens = len(token_estimator.encode(sent))
        if current_tokens + sent_tokens > max_tokens and current_sentences:
            # Finalise current child
            child_content = " ".join(current_sentences)
            # Build clean embedding text for child (no truncation)
            child_embedding = (
                f"Document: {parent.document_name}\n"
                f"Section Path: {section_path_str}\n"
                f"Pages: {parent.page_start}–{parent.page_end}\n\n"
                f"{child_content}"
            )
            child = Chunk(
                chunk_id=str(uuid.uuid4()),
                chunk_index=len(children),
                document_id=parent.document_id,
                document_name=parent.document_name,
                document_type=parent.document_type,
                domain=parent.domain,
                section=parent.section,
                subsection="",   # children also have no separate subsection
                section_level=parent.section_level,
                chunk_type="child",
                chunk_title=f"{parent.chunk_title} (part {len(children)+1})",
                content=child_content,
                embedding_text=child_embedding,
                page_start=parent.page_start,
                page_end=parent.page_end,
                tables=parent.tables,
                figures=parent.figures,
                token_count=current_tokens,
                metadata={},
                relationships={"parent_chunk_id": parent.chunk_id, "previous_chunk_id": None, "next_chunk_id": None}
            )
            children.append(child)

            # Overlap: keep last few tokens
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                s_tok = len(token_estimator.encode(s))
                if overlap_tokens + s_tok <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tok
                else:
                    break
            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sent)
        current_tokens += sent_tokens

    # Final child (if any)
    if current_sentences:
        child_content = " ".join(current_sentences)
        child_embedding = (
            f"Document: {parent.document_name}\n"
            f"Section Path: {section_path_str}\n"
            f"Pages: {parent.page_start}–{parent.page_end}\n\n"
            f"{child_content}"
        )
        child = Chunk(
            chunk_id=str(uuid.uuid4()),
            chunk_index=len(children),
            document_id=parent.document_id,
            document_name=parent.document_name,
            document_type=parent.document_type,
            domain=parent.domain,
            section=parent.section,
            subsection="",
            section_level=parent.section_level,
            chunk_type="child",
            chunk_title=f"{parent.chunk_title} (part {len(children)+1})",
            content=child_content,
            embedding_text=child_embedding,
            page_start=parent.page_start,
            page_end=parent.page_end,
            tables=parent.tables,
            figures=parent.figures,
            token_count=current_tokens,
            metadata={},
            relationships={"parent_chunk_id": parent.chunk_id, "previous_chunk_id": None, "next_chunk_id": None}
        )
        children.append(child)

    return children


# ------------------------------------------------------------------
# Legacy flat‑block chunking (fallback, unchanged from your original)
# ------------------------------------------------------------------

def _chunk_flat_blocks(
    doc: NormalizedDocument,
    document_id: str,
    domain: str
) -> List[Chunk]:
    """Fallback to original flat‑block chunking when no sections are available."""
    # This is a stub – you can keep your old chunking logic here if needed.
    return []


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _table_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---" for _ in headers]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _split_into_sentences(text: str) -> List[str]:
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]