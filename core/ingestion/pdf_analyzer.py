"""
core/ingestion/pdf_analyzer.py

Analyzes a PDF and produces:
- NormalizedDocument with described images (Gemini/Gemma)
- Detailed per‑page classification (text, images, tables, blank)
- Aggregated summary text
- All image descriptions are included in the page description
- Uses layout model (unstructured) for digital pages too, to reliably detect images/tables.
"""

import os
import io
from typing import Dict, List, Any
from PIL import Image
import fitz
from dotenv import load_dotenv

load_dotenv()

from core.schemas.models import (
    ContentBlock, TableBlock, ImageBlock,
    BlockType, NormalizedDocument, Chunk
)
from core.ingestion.pdf_detector import detect_pdf_type
from core.ingestion.handlers.digital_handler import (
    process_text_block, process_image_block, extract_tables_from_page
)
from core.ingestion.handlers.scanned_handler import (
    analyze_layout, merge_layout_and_ocr,
    extract_text_with_ocr
)
from core.ingestion.gemma_client import describe_image_with_gemma
from core.ingestion.chunker import chunk_document


def analyze_pdf(file_path: str, use_gemma: bool = False) -> Dict[str, Any]:
    """
    Full analysis: returns extracted document, chunks, detailed per‑page classification and summary text.
    """
    file_name = os.path.basename(file_path)
    pdf_type, page_types = detect_pdf_type(file_path)
    doc = fitz.open(file_path)
    total_pages = len(page_types)

    all_blocks = []
    detailed_summary = []

    for page_num in range(total_pages):
        page = doc[page_num]
        page_number = page_num + 1
        page_type = page_types[page_num]

        has_text = False
        has_image = False
        has_table = False
        is_blank = True
        page_headings = []
        page_paragraphs = []
        gemma_descriptions = []

        all_blocks.append(ContentBlock(
            type=BlockType.PAGE_BREAK,
            text=f"--- Page {page_number} ---",
            page=page_number
        ))

        if page_type == "digital":
            # ---------- 1. Text extraction via PyMuPDF ----------
            blocks_data = page.get_text("dict")["blocks"]
            for block in blocks_data:
                if block["type"] == 0:  # text
                    text_blocks = process_text_block(block, page_number, 0)
                    for tb in text_blocks:
                        if tb.text.strip():
                            has_text = True
                            is_blank = False
                            if tb.type == BlockType.HEADING:
                                page_headings.append(tb.text)
                            else:
                                page_paragraphs.append(tb.text)
                        all_blocks.append(tb)
                # We ignore block["type"] == 1 (images) because we'll use the layout model later

            # ---------- 2. Tables via PyMuPDF (fallback) ----------
            table_blocks = extract_tables_from_page(doc, page_num)
            if table_blocks:
                has_table = True
                is_blank = False
                all_blocks.extend(table_blocks)

            # ---------- 3. Layout model for images & tables ----------
            pix = page.get_pixmap(dpi=200)   # moderate DPI for speed
            page_image = Image.open(io.BytesIO(pix.tobytes("png")))

            layout_elements = analyze_layout(page_image, page_number)
            for lb in layout_elements:
                if lb.type == BlockType.TABLE:
                    # Add only if no table was already detected by PyMuPDF
                    if not table_blocks:
                        has_table = True
                        is_blank = False
                        all_blocks.append(lb)
                elif lb.type == BlockType.IMAGE:
                    has_image = True
                    is_blank = False
                    if lb.image and use_gemma:
                        if lb.image.image_bytes and not lb.image.description:
                            desc = describe_image_with_gemma(lb.image.image_bytes)
                            if desc:
                                lb.image.description = desc
                                lb.text = desc
                                gemma_descriptions.append(desc)
                    all_blocks.append(lb)

            # ---------- 4. Fallback: all embedded images via page.get_images() ----------
            page_images = page.get_images()
            if page_images:
                has_image = True
                is_blank = False
                processed_xrefs = set()
                for img_info in page_images:
                    xref = img_info[0]
                    if xref in processed_xrefs:
                        continue
                    processed_xrefs.add(xref)
                    try:
                        base_image = doc.extract_image(xref)
                        img_bytes = base_image["image"]
                        mime = f"image/{base_image['ext']}"
                    except Exception:
                        continue

                    # Avoid duplicates (simple check by comparing first 20 bytes)
                    already = False
                    for blk in all_blocks:
                        if blk.type == BlockType.IMAGE and blk.image and blk.image.image_bytes:
                            if blk.image.image_bytes[:20] == img_bytes[:20]:
                                already = True
                                break
                    if not already:
                        description = None
                        if use_gemma:
                            description = describe_image_with_gemma(img_bytes)
                        if description:
                            gemma_descriptions.append(description)
                        image_block = ImageBlock(
                            image_bytes=img_bytes,
                            mime_type=mime,
                            description=description,
                            page=page_number
                        )
                        all_blocks.append(ContentBlock(
                            type=BlockType.IMAGE,
                            text=description or "",
                            page=page_number,
                            image=image_block
                        ))

        else:  # scanned page
            pix = page.get_pixmap(dpi=300)
            page_image = Image.open(io.BytesIO(pix.tobytes("png")))

            layout_blocks = analyze_layout(page_image, page_number)
            ocr_text_blocks = extract_text_with_ocr(page_image, page_number)
            merged = merge_layout_and_ocr(layout_blocks, ocr_text_blocks, page_number)

            for block in merged:
                if block.type == BlockType.PARAGRAPH and block.text.strip():
                    has_text = True
                    is_blank = False
                    page_paragraphs.append(block.text)
                elif block.type == BlockType.TABLE:
                    has_table = True
                    is_blank = False
                elif block.type == BlockType.IMAGE:
                    has_image = True
                    is_blank = False
                    if block.image and block.image.description:
                        gemma_descriptions.append(block.image.description)
                all_blocks.append(block)

        # Build page description
        description = ""
        if gemma_descriptions:
            description = "\n---\n".join(gemma_descriptions)
        elif has_text:
            first_heading = page_headings[0].strip() if page_headings else ""
            unique_paragraphs = [p for p in page_paragraphs if p.strip() != first_heading]
            if unique_paragraphs:
                description = unique_paragraphs[0][:200]
            elif page_paragraphs:
                description = page_paragraphs[0][:200]
            else:
                description = "Text content"
        elif has_image:
            description = "Image(s) detected"
        elif has_table:
            description = "Table(s) detected"
        elif is_blank:
            description = "Blank page"
        else:
            description = "No content"

        detailed_summary.append({
            "page": page_number,
            "type": page_type,
            "digital_text": "✅" if has_text else "❌",
            "image": "✅" if has_image else "❌",
            "table": "✅" if has_table else "❌",
            "blank": "✅" if is_blank else "❌",
            "description": description
        })

    doc.close()

    normalized_doc = NormalizedDocument(
        file_name=file_name,
        total_pages=total_pages,
        pdf_type=pdf_type,
        blocks=all_blocks
    )

    chunks = chunk_document(normalized_doc)
    summary_text = generate_summary(detailed_summary)

    return {
        "file_name": file_name,
        "pdf_type": pdf_type,
        "total_pages": total_pages,
        "total_chunks": len(chunks),
        "detailed_summary": detailed_summary,
        "summary_text": summary_text,
        "chunks": [c.to_dict() for c in chunks]
    }


def generate_summary(page_data: List[Dict]) -> str:
    """Create a formatted summary string (aggregated ranges)."""
    digital_pages = []
    scanned_pages = []
    image_pages = []
    table_pages = []
    blank_pages = []

    for p in page_data:
        if p["type"] == "digital":
            digital_pages.append(p["page"])
        elif p["type"] == "scanned":
            scanned_pages.append(p["page"])
        if p["image"] == "✅":
            image_pages.append(p["page"])
        if p["table"] == "✅":
            table_pages.append(p["page"])
        if p["blank"] == "✅":
            blank_pages.append(p["page"])

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