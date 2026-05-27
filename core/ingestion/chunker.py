"""
core/ingestion/chunker.py

Takes a NormalizedDocument and returns smart chunks ready for embedding.
Chunking strategy based on RAG failure research:
- Keep tables whole (never split them)
- Merge cross‑page paragraphs
- Merge small consecutive paragraphs (<100 chars)
- Skip chunks that are only empty image placeholders
- Add context headers to every chunk
- Target 500–1000 chars for embedding_text
"""

import uuid
from typing import List, Optional
from core.schemas.models import (
    NormalizedDocument, ContentBlock, Chunk,
    BlockType
)


def chunk_document(doc: NormalizedDocument) -> List[Chunk]:
    """
    Convert a NormalizedDocument into a list of Chunks ready for embedding.
    
    Strategy:
    1. Merge cross‑page paragraphs that got split
    2. Merge small consecutive paragraphs (<100 chars)
    3. Group blocks into logical segments
    4. Create self‑contained chunks with context headers
    """
    if not doc.blocks:
        return []

    # Step 1: Merge cross‑page paragraphs
    merged_blocks = merge_cross_page_paragraphs(doc.blocks)

    # Step 2: Merge small consecutive paragraphs
    merged_blocks = merge_small_paragraphs(merged_blocks)

    # Step 3: Group blocks into logical segments
    segments = group_into_segments(merged_blocks)

    # Step 4: Create chunks from segments
    chunks = []
    for i, segment in enumerate(segments):
        chunk = segment_to_chunk(
            segment=segment,
            chunk_index=i,
            document_name=doc.file_name,
            pdf_type=doc.pdf_type
        )
        if chunk and not is_empty_image_placeholder(chunk):
            chunks.append(chunk)

    return chunks


def merge_cross_page_paragraphs(blocks: List[ContentBlock]) -> List[ContentBlock]:
    """
    Merge paragraphs that were split across page boundaries.
    Checks if a paragraph at end of one page continues on the next.
    """
    if len(blocks) < 3:
        return blocks

    merged = []
    i = 0

    while i < len(blocks):
        block = blocks[i]

        if (block.type == BlockType.PARAGRAPH and
            i + 2 < len(blocks) and
            blocks[i + 1].type == BlockType.PAGE_BREAK and
            blocks[i + 2].type == BlockType.PARAGRAPH):

            current_text = block.text.rstrip()
            next_text = blocks[i + 2].text.lstrip()

            # Merge if sentence doesn't end with punctuation
            if current_text and not current_text[-1] in '.!?':
                merged_text = current_text + " " + next_text
                merged_block = ContentBlock(
                    type=BlockType.PARAGRAPH,
                    text=merged_text,
                    page=block.page,
                    section_id=block.section_id,
                    metadata={
                        **block.metadata,
                        "merged_pages": f"{block.page}-{blocks[i + 2].page}"
                    }
                )
                merged.append(merged_block)
                i += 3
                continue

        merged.append(block)
        i += 1

    return merged


def merge_small_paragraphs(blocks: List[ContentBlock]) -> List[ContentBlock]:
    """
    Merge consecutive PARAGRAPH blocks that are shorter than 100 characters
    into a single combined paragraph. This reduces tiny, low‑value chunks.
    """
    if not blocks:
        return []

    merged = []
    buffer = []

    for block in blocks:
        if block.type == BlockType.PARAGRAPH and len(block.text) < 100:
            buffer.append(block)
        else:
            if buffer:
                combined_text = " ".join(b.text for b in buffer)
                merged.append(ContentBlock(
                    type=BlockType.PARAGRAPH,
                    text=combined_text,
                    page=buffer[0].page,
                    section_id=buffer[0].section_id,
                    metadata={"merged_paragraphs": len(buffer)}
                ))
                buffer = []
            merged.append(block)

    if buffer:
        combined_text = " ".join(b.text for b in buffer)
        merged.append(ContentBlock(
            type=BlockType.PARAGRAPH,
            text=combined_text,
            page=buffer[0].page,
            section_id=buffer[0].section_id,
            metadata={"merged_paragraphs": len(buffer)}
        ))

    return merged


def group_into_segments(blocks: List[ContentBlock]) -> List[List[ContentBlock]]:
    """
    Group blocks into logical segments for chunking.
    
    A new segment starts when:
    - A new heading appears
    - A table is encountered
    - An image block appears
    """
    if not blocks:
        return []

    segments = []
    current_segment = []

    for block in blocks:
        # Skip page breaks for segment grouping
        if block.type == BlockType.PAGE_BREAK:
            continue

        # New heading = new segment
        if block.type == BlockType.HEADING:
            if current_segment:
                segments.append(current_segment)
            current_segment = [block]
            continue

        # Tables get their own segment
        if block.type == BlockType.TABLE:
            if current_segment:
                segments.append(current_segment)
            segments.append([block])
            current_segment = []
            continue

        # Images get their own segment
        if block.type == BlockType.IMAGE:
            if current_segment:
                segments.append(current_segment)
            segments.append([block])
            current_segment = []
            continue

        # Regular content – add to current segment
        current_segment.append(block)

    if current_segment:
        segments.append(current_segment)

    return segments


def segment_to_chunk(
    segment: List[ContentBlock],
    chunk_index: int,
    document_name: str,
    pdf_type: str
) -> Optional[Chunk]:
    """
    Convert a segment of blocks into a single Chunk.
    """
    if not segment:
        return None

    chunk_type = determine_chunk_type(segment)
    section_path = build_section_path(segment)
    content = build_content(segment)

    if not content.strip():
        return None

    pages = [b.page for b in segment]
    page_start = min(pages)
    page_end = max(pages)

    embedding_text = f"""Document: {document_name}
Section: {section_path}
Pages: {page_start}-{page_end}

{content}"""

    if len(embedding_text.strip()) < 50:
        return None

    chunk_id = str(uuid.uuid4())

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
        metadata={
            "pdf_type": pdf_type,
            "block_count": len(segment)
        }
    )


def determine_chunk_type(segment: List[ContentBlock]) -> str:
    """Determine the type of chunk from its blocks."""
    types = [b.type for b in segment]

    if BlockType.TABLE in types:
        return "table"
    if BlockType.IMAGE in types:
        return "image_description"
    if BlockType.HEADING in types and len(segment) == 1:
        return "heading"
    if len(types) > 1:
        return "mixed"
    return "paragraph"


def build_section_path(segment: List[ContentBlock]) -> str:
    """Build section path string from heading blocks in the segment."""
    headings = [b.text for b in segment if b.type == BlockType.HEADING]

    if headings:
        return " > ".join(headings)

    for block in segment:
        if block.section_id:
            return block.section_id

    return "General"


def build_content(segment: List[ContentBlock]) -> str:
    """Build content string from segment blocks."""
    parts = []

    for block in segment:
        if block.type == BlockType.TABLE and block.table:
            parts.append(table_to_markdown(
                block.table.headers,
                block.table.rows
            ))
        elif block.type == BlockType.IMAGE and block.image:
            if block.image.description:
                parts.append(block.image.description)
            elif block.image.caption:
                parts.append(f"[Image: {block.image.caption}]")
            else:
                parts.append("[Image attached]")
        elif block.type == BlockType.HEADING:
            level = block.level or 1
            prefix = "#" * level
            parts.append(f"{prefix} {block.text}")
        else:
            if block.text.strip():
                parts.append(block.text)

    return "\n\n".join(parts)


def is_empty_image_placeholder(chunk: Chunk) -> bool:
    """
    Returns True if the chunk is just an image placeholder
    with no real description (e.g., only '[Image attached]').
    """
    return (chunk.chunk_type == "image_description" and
            chunk.content.strip() in ("[Image attached]", ""))


def table_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    """Convert table headers and rows to markdown format."""
    lines = []

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---" for _ in headers]) + "|")

    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)