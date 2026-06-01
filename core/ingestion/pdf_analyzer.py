"""
core/ingestion/pdf_analyzer.py

Analyzes a PDF and produces:
- NormalizedDocument with described images (Gemini)
- Detailed per‑page classification (text, images, tables, blank)
- Aggregated summary text
- Chunks in the new detailed JSON format (with document_id, section, tables, etc.)
"""

import os
import uuid
import hashlib
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from core.schemas.models import (
    ContentBlock, TableBlock, ImageBlock,
    BlockType, NormalizedDocument, Chunk
)
from core.ingestion.pdf_detector import detect_pdf_type
from core.ingestion.handlers.digital_handler import extract_digital
from core.ingestion.handlers.mixed_handler import extract_mixed
from core.ingestion.gemma_client import describe_image_with_gemma
from core.ingestion.chunker import chunk_document

_gemma_cache: Dict[str, str] = {}


def _get_image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def _describe_image_cached(image_bytes: bytes) -> Optional[str]:
    img_hash = _get_image_hash(image_bytes)
    if img_hash in _gemma_cache:
        print("      (cached) Using previously obtained description")
        return _gemma_cache[img_hash]
    desc = describe_image_with_gemma(image_bytes)
    if desc:
        _gemma_cache[img_hash] = desc
    return desc


def analyze_pdf(
    file_path: str,
    use_gemma: bool = False,
    document_id: Optional[str] = None,
    domain: str = "general"
) -> Dict[str, Any]:
    """
    Analyze a PDF and return structured results.

    Args:
        file_path: Path to the PDF file.
        use_gemma: Whether to generate image descriptions via Gemini.
        document_id: Unique ID for the document (generated if not provided).
        domain: Domain label (default "general").

    Returns:
        Dictionary with keys: file_name, pdf_type, total_pages, total_chunks,
        detailed_summary, summary_text, chunks.
    """
    file_name = os.path.basename(file_path)
    pdf_type, page_types = detect_pdf_type(file_path)

    if not document_id:
        document_id = str(uuid.uuid4())

    # Route to appropriate handler
    if pdf_type == "digital":
        normalized_doc = extract_digital(file_path, use_gemma=use_gemma)
    else:
        normalized_doc = extract_mixed(file_path, use_gemma=use_gemma)

    if normalized_doc is None:
        return {
            "file_name": file_name,
            "pdf_type": pdf_type,
            "total_pages": 0,
            "total_chunks": 0,
            "detailed_summary": [],
            "summary_text": "Unsupported PDF type",
            "chunks": []
        }

    # ------------------------------------------------------------------
    # Build detailed page summary from the NormalizedDocument
    # ------------------------------------------------------------------
    page_text_flags = {}
    page_image_flags = {}
    page_table_flags = {}
    page_descriptions = {}
    page_metrics = {}

    def process_section(section):
        # Tables
        for tbl in section.tables:
            p = tbl.page
            page_table_flags[p] = True
            if p not in page_descriptions:
                page_descriptions[p] = f"Table: {', '.join(tbl.headers)}"
        # Figures
        for fig in section.figures:
            p = fig.page
            page_image_flags[p] = True
            if p not in page_descriptions:
                desc = fig.description or fig.caption or "Image"
                page_descriptions[p] = desc[:200]
        # Text content
        if section.content:
            for p in range(section.page_start, section.page_end + 1):
                page_text_flags[p] = True
                if p not in page_descriptions:
                    page_descriptions[p] = section.content[:200]
        for child in section.children:
            process_section(child)

    if normalized_doc.sections:
        for section in normalized_doc.sections:
            process_section(section)

    # Legacy blocks fallback
    for block in normalized_doc.blocks:
        p = block.page
        if block.type in (BlockType.PARAGRAPH, BlockType.HEADING):
            page_text_flags[p] = True
            if p not in page_descriptions:
                page_descriptions[p] = block.text[:200]
        elif block.type == BlockType.IMAGE:
            page_image_flags[p] = True
            if block.text and p not in page_descriptions:
                page_descriptions[p] = block.text[:200]
        elif block.type == BlockType.TABLE:
            page_table_flags[p] = True
            if p not in page_descriptions:
                page_descriptions[p] = "Table: " + (block.text[:100] if block.text else "")
        elif block.type == BlockType.PAGE_BREAK:
            meta = block.metadata
            if meta:
                page_metrics[p] = {
                    "has_text": meta.get("has_text", False),
                    "text_length": meta.get("text_length", 0),
                    "raster_images": meta.get("raster_images", 0),
                    "vector_drawings": meta.get("vector_drawings", 0),
                    "annotations": meta.get("annotations", 0),
                    "links": meta.get("links", 0),
                    "image_coverage_ratio": meta.get("image_coverage_ratio", 0.0)
                }

    detailed_summary = []
    for page_num in range(1, normalized_doc.total_pages + 1):
        metrics = page_metrics.get(page_num, {})
        summary_entry = {
            "page": page_num,
            "type": pdf_type,
            "digital_text": "✅" if page_text_flags.get(page_num) else "❌",
            "image": "✅" if page_image_flags.get(page_num) else "❌",
            "table": "✅" if page_table_flags.get(page_num) else "❌",
            "blank": "✅" if not (page_text_flags.get(page_num) or page_image_flags.get(page_num) or page_table_flags.get(page_num)) else "❌",
            "description": page_descriptions.get(page_num, "")
        }
        if metrics:
            summary_entry.update(metrics)
        detailed_summary.append(summary_entry)

    # Chunk the document
    chunks = chunk_document(
        normalized_doc,
        document_id=document_id,
        domain=domain
    )

    # ------------------------------------------------------------------
    # ✨ NEW: Enrich each chunk’s figures with a compressed base64 image
    #        so the frontend can display the actual image + description.
    # ------------------------------------------------------------------
    _enrich_chunks_with_images(chunks, normalized_doc)

    summary_text = generate_summary(detailed_summary)

    return {
        "file_name": file_name,
        "pdf_type": pdf_type,
        "total_pages": normalized_doc.total_pages,
        "total_chunks": len(chunks),
        "detailed_summary": detailed_summary,
        "summary_text": summary_text,
        "chunks": [c.to_dict() for c in chunks]
    }


def _enrich_chunks_with_images(chunks: List[Chunk], doc: NormalizedDocument):
    """
    Walk through all ImageBlock objects in the document (sections tree)
    and inject a 'base64' field into every chunk figure that matches
    on page + description.
    """
    # Build a map: (page, description) -> ImageBlock (for quick lookup)
    image_lookup: Dict[tuple, ImageBlock] = {}

    def collect_images(section):
        for fig in section.figures:
            # Use (page, description) as key; empty description is fine
            key = (fig.page, fig.description or fig.caption or "")
            image_lookup[key] = fig
        for child in section.children:
            collect_images(child)

    for section in doc.sections:
        collect_images(section)

    # For each chunk, try to match each figure and add base64
    for chunk in chunks:
        updated_figures = []
        for fig_dict in chunk.figures:
            page = fig_dict.get("page", 0)
            desc = fig_dict.get("description", "") or ""
            key = (page, desc)
            img_block = image_lookup.get(key)
            if img_block:
                # Add the display base64 (compressed JPEG)
                fig_dict["base64"] = img_block.to_display_base64(max_width=300)
            updated_figures.append(fig_dict)
        chunk.figures = updated_figures


def generate_summary(page_data: List[Dict]) -> str:
    digital_pages = [p["page"] for p in page_data if p["type"] == "digital"]
    scanned_pages = [p["page"] for p in page_data if p["type"] == "scanned"]
    image_pages = [p["page"] for p in page_data if p["image"] == "✅"]
    table_pages = [p["page"] for p in page_data if p["table"] == "✅"]
    blank_pages = [p["page"] for p in page_data if p["blank"] == "✅"]

    def format_ranges(pages):
        if not pages:
            return "None detected"
        ranges = []
        start = pages[0]
        end = start
        for num in pages[1:]:
            if num == end + 1:
                end = num
            else:
                ranges.append((start, end))
                start = num
                end = num
        ranges.append((start, end))
        return ", ".join([f"{s}" if s == e else f"{s}–{e}" for s, e in ranges])

    lines = [
        "Summary",
        "Type\tPages",
        f"Digital text pages\t{format_ranges(digital_pages)}",
        f"Scanned pages\t{format_ranges(scanned_pages)}",
        f"Pages containing images/diagrams\t{format_ranges(image_pages)}",
        f"Pages containing tables\t{format_ranges(table_pages)}",
        f"Blank pages\t{format_ranges(blank_pages)}"
    ]
    return "\n".join(lines)