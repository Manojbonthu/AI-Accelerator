"""
core/ingestion/chunker.py

Takes a NormalizedDocument with hierarchical sections and returns chunks ready for embedding.
Only child chunks are created – parent chunks are used purely for structure.
Tiny sections are merged to avoid single‑line chunks.
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
    max_child_tokens: int = 700,      # increased from 500
    overlap_tokens: int = 50,
    min_section_content_length: int = 100   # sections shorter than this get merged
) -> List[Chunk]:
    """
    Convert a NormalizedDocument into child‑only chunks using hierarchical sections.

    - Parents are never embedded (they are kept as metadata only).
    - Very small sections (e.g., headings with barely any text) are merged into the next sibling.
    - Each child chunk is ≤ max_child_tokens tokens.
    """
    if not document_id:
        document_id = str(uuid.uuid4())

    if doc.sections:
        # ---- 1. Merge tiny sections to avoid orphan chunks ----
        merged_sections = _merge_tiny_sections(doc.sections, min_section_content_length)

        # ---- 2. Create parent chunks (for structure, NOT for embedding) ----
        parents = _create_parent_chunks_from_sections(
            merged_sections, doc, document_id, domain
        )

        # ---- 3. Split each parent into child chunks (only children go to output) ----
        all_chunks: List[Chunk] = []
        for parent in parents:
            children = _split_parent_into_children(parent, max_child_tokens, overlap_tokens)
            all_chunks.extend(children)

        # ---- 4. Link children sequentially (preserve parent_chunk_id) ----
        for i, chunk in enumerate(all_chunks):
            if i > 0:
                chunk.relationships["previous_chunk_id"] = all_chunks[i-1].chunk_id
            if i < len(all_chunks) - 1:
                chunk.relationships["next_chunk_id"] = all_chunks[i+1].chunk_id

        return all_chunks

    elif doc.blocks:
        return _chunk_flat_blocks(doc, document_id, domain)
    else:
        return []


# ------------------------------------------------------------------
# Merging tiny sections (noise reduction)
# ------------------------------------------------------------------

def _merge_tiny_sections(
    sections: List[Section],
    min_len: int = 100
) -> List[Section]:
    """
    Walk through sibling sections. If a section has less than min_len characters
    of own content (not counting children), append its content to the next sibling
    that has enough content, or keep as is if it's the last one.
    """
    merged = []
    buffer = ""           # accumulated tiny content
    buffer_start = None
    buffer_end = None

    def flush_buffer(next_section: Section):
        nonlocal buffer, buffer_start, buffer_end
        if not buffer:
            return
        # Prepend the buffered tiny content to the next section's content
        next_section.content = buffer + "\n\n" + next_section.content
        if buffer_start is not None and buffer_start < next_section.page_start:
            next_section.page_start = buffer_start
        buffer = ""
        buffer_start = None
        buffer_end = None

    for sec in sections:
        own_content = sec.content.strip()
        # Recursively merge children first
        if sec.children:
            sec.children = _merge_tiny_sections(sec.children, min_len)

        # Check if section itself is tiny (no content of its own, or very short)
        if len(own_content) < min_len:
            # Accumulate content for later merge
            if not buffer:
                buffer_start = sec.page_start
            buffer_end = sec.page_end
            # Keep the text (the heading itself is part of the content)
            if own_content:
                buffer = (buffer + "\n" + own_content).strip() if buffer else own_content
            # Don't add this section yet; we'll merge it into the next one
            continue
        else:
            # This section is large enough. First flush any buffered tiny content into it.
            if buffer:
                sec.content = buffer + "\n\n" + sec.content
                if buffer_start is not None and buffer_start < sec.page_start:
                    sec.page_start = buffer_start
                buffer = ""
                buffer_start = None
                buffer_end = None
            merged.append(sec)

    # If there's remaining buffer at the end, turn it into a final section
    if buffer:
        # Create a synthetic section with the leftover tiny content
        fake_section = Section(
            title="(Merged)",
            level=1,
            section_path=["(Merged)"],
            page_start=buffer_start or 0,
            page_end=buffer_end or 0,
            content=buffer,
            tables=[],
            figures=[],
            children=[]
        )
        merged.append(fake_section)

    return merged


# ------------------------------------------------------------------
# Section‑based parent creation (not embedded)
# ------------------------------------------------------------------

def _create_parent_chunks_from_sections(
    sections: List[Section],
    doc: NormalizedDocument,
    document_id: str,
    domain: str,
    idx_start: int = 0
) -> List[Chunk]:
    """Recursively traverse section tree and create one parent chunk per section (for metadata only)."""
    token_estimator = tiktoken.get_encoding("cl100k_base")
    chunks = []
    global_idx = idx_start
    for section in sections:
        full_content = section.content.strip()
        if not full_content and not section.children:
            # Empty leaf – skip
            continue

        # Build embedding text (not actually embedded, but kept for consistency)
        section_path_str = " > ".join(section.section_path)
        embedding_text = (
            f"Document: {doc.file_name}\n"
            f"Section Path: {section_path_str}\n"
            f"Pages: {section.page_start}–{section.page_end}\n\n"
            f"{full_content}"
        )

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

        token_count = len(token_estimator.encode(full_content))

        parent = Chunk(
            chunk_id=str(uuid.uuid4()),
            chunk_index=global_idx,
            document_id=document_id,
            document_name=doc.file_name,
            document_type=doc.pdf_type,
            domain=domain,
            section=section.title,
            subsection="",
            section_level=section.level,
            chunk_type="parent",
            chunk_title=section.title,
            content=full_content,
            embedding_text=embedding_text,
            page_start=section.page_start,
            page_end=section.page_end,
            tables=tables_dict,
            figures=figures_dict,
            token_count=token_count,
            metadata={"section_path": section.section_path}
        )
        chunks.append(parent)
        global_idx += 1

        # Process children recursively
        child_chunks = _create_parent_chunks_from_sections(
            section.children, doc, document_id, domain, global_idx
        )
        chunks.extend(child_chunks)
        global_idx += len(child_chunks)

    return chunks


def _split_parent_into_children(parent: Chunk, max_tokens: int, overlap: int) -> List[Chunk]:
    """Split a parent chunk's content into child chunks (only children are embedded)."""
    sentences = _split_into_sentences(parent.content)
    if not sentences:
        return []

    token_estimator = tiktoken.get_encoding("cl100k_base")
    children = []
    current_sentences = []
    current_tokens = 0

    section_path = parent.metadata.get("section_path", [])
    section_path_str = " > ".join(section_path)

    def make_child(sents, token_count):
        content = " ".join(sents)
        return Chunk(
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
            content=content,
            embedding_text=(
                f"Document: {parent.document_name}\n"
                f"Section Path: {section_path_str}\n"
                f"Pages: {parent.page_start}–{parent.page_end}\n\n"
                f"{content}"
            ),
            page_start=parent.page_start,
            page_end=parent.page_end,
            tables=parent.tables,
            figures=parent.figures,
            token_count=token_count,
            metadata={},
            relationships={
                "parent_chunk_id": parent.chunk_id,
                "previous_chunk_id": None,
                "next_chunk_id": None
            }
        )

    for sent in sentences:
        sent_tokens = len(token_estimator.encode(sent))
        if current_tokens + sent_tokens > max_tokens and current_sentences:
            children.append(make_child(current_sentences, current_tokens))

            # Overlap: keep last few sentences
            overlap_sents = []
            overlap_tok = 0
            for s in reversed(current_sentences):
                s_tok = len(token_estimator.encode(s))
                if overlap_tok + s_tok <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_tok += s_tok
                else:
                    break
            current_sentences = overlap_sents
            current_tokens = overlap_tok

        current_sentences.append(sent)
        current_tokens += sent_tokens

    if current_sentences:
        children.append(make_child(current_sentences, current_tokens))

    return children


# ------------------------------------------------------------------
# Legacy flat‑block chunking (unchanged)
# ------------------------------------------------------------------

def _chunk_flat_blocks(
    doc: NormalizedDocument,
    document_id: str,
    domain: str
) -> List[Chunk]:
    """Fallback – empty for now (you can keep your existing logic)."""
    return []


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _split_into_sentences(text: str) -> List[str]:
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _table_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---" for _ in headers]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)