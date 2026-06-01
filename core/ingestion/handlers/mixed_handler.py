"""
core/ingestion/handlers/mixed_handler.py

Mixed PDF handler – processes each page individually.
Digital pages → use PyMuPDF extraction.
Scanned pages → use OCR + YOLO + optional Gemma.
"""

import os
import io
import fitz
from typing import List

from core.schemas.models import (
    ContentBlock, TableBlock, ImageBlock,
    BlockType, NormalizedDocument, Section
)
from core.ingestion.pdf_detector import detect_pdf_type
from core.ingestion.handlers.digital_handler import extract_digital_page_content
from core.ingestion.handlers.scanned_handler import extract_scanned_page_content
from core.ingestion.gemma_client import describe_image_with_gemma


def extract_mixed(file_path: str, use_gemma: bool = True) -> NormalizedDocument:
    """
    Process a mixed PDF page‑by‑page.
    Each page is classified as digital or scanned, then routed to the appropriate handler.
    """
    file_name = os.path.basename(file_path)
    doc = fitz.open(file_path)
    total_pages = len(doc)
    
    # Get per‑page classification
    _, page_types = detect_pdf_type(file_path)
    
    all_blocks: List[ContentBlock] = []
    
    for page_num in range(total_pages):
        page = doc[page_num]
        page_number = page_num + 1
        page_type = page_types[page_num]
        
        print(f"Processing page {page_number} ({page_type})")
        
        # Add page break marker
        all_blocks.append(ContentBlock(
            type=BlockType.PAGE_BREAK,
            text=f"--- Page {page_number} ---",
            page=page_number,
            metadata={"document_name": file_name}
        ))
        
        if page_type == "digital":
            # Use digital extraction for this page
            blocks = extract_digital_page_content(page, page_number, doc, use_gemma)
        else:
            # Use scanned extraction for this page
            blocks = extract_scanned_page_content(page, page_number, use_gemma)
        
        all_blocks.extend(blocks)
    
    doc.close()
    
    # Build section hierarchy from blocks
    sections = build_sections_from_blocks(all_blocks, total_pages)
    
    return NormalizedDocument(
        file_name=file_name,
        total_pages=total_pages,
        pdf_type="mixed",
        sections=sections,
        blocks=all_blocks
    )


def extract_digital_page_content(page: fitz.Page, page_num: int, doc: fitz.Document, use_gemma: bool) -> List[ContentBlock]:
    """
    Extract content from a single digital page.
    Uses the same logic as digital_handler but for one page only.
    """
    blocks = []
    
    # Get text, images, tables from the page
    text_dict = page.get_text("dict")
    for block in text_dict["blocks"]:
        if block["type"] == 0:  # text
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    font_size = span["size"]
                    font = span.get("font", "")
                    is_bold = "Bold" in font or "bold" in font.lower()
                    
                    if font_size >= 14 or (font_size >= 12 and is_bold):
                        level = 1 if font_size >= 18 else 2
                        blocks.append(ContentBlock(
                            type=BlockType.HEADING,
                            text=text,
                            level=level,
                            page=page_num,
                            metadata={"source": "digital"}
                        ))
                    else:
                        blocks.append(ContentBlock(
                            type=BlockType.PARAGRAPH,
                            text=text,
                            page=page_num,
                            metadata={"source": "digital"}
                        ))
        elif block["type"] == 1:  # image
            # Extract image bytes
            try:
                xref = block.get("image", 0)
                if xref:
                    img_dict = doc.extract_image(xref)
                    img_bytes = img_dict["image"]
                    image_block = ImageBlock(
                        image_bytes=img_bytes,
                        mime_type="image/png",
                        page=page_num,
                        confidence=1.0
                    )
                    if use_gemma:
                        desc = describe_image_with_gemma(img_bytes)
                        if desc:
                            image_block.description = desc
                    blocks.append(ContentBlock(
                        type=BlockType.IMAGE,
                        text=image_block.description or "[Image]",
                        page=page_num,
                        image=image_block,
                        metadata={"source": "digital"}
                    ))
            except Exception:
                pass
    
    # Extract tables
    tables = page.find_tables()
    for table in tables.tables:
        if not table.header or not table.extract():
            continue
        headers = [str(h) for h in table.header.names]
        rows = [[str(cell) for cell in row] for row in table.extract()]
        table_block = TableBlock(headers=headers, rows=rows, page=page_num)
        # Create markdown representation
        md = "| " + " | ".join(headers) + " |\n"
        md += "|" + "|".join(["---"] * len(headers)) + "|\n"
        for row in rows:
            md += "| " + " | ".join(row) + " |\n"
        blocks.append(ContentBlock(
            type=BlockType.TABLE,
            text=md,
            page=page_num,
            table=table_block,
            metadata={"source": "digital"}
        ))
    
    return blocks


def extract_scanned_page_content(page: fitz.Page, page_num: int, use_gemma: bool) -> List[ContentBlock]:
    """
    Extract content from a single scanned page.
    Uses OCR + YOLO + optional Gemma (reuses logic from scanned_handler).
    """
    from core.ingestion.handlers.scanned_handler import (
        page_to_pil, extract_ocr_text, detect_meaningful_images,
        parse_ocr_text_to_blocks, describe_image_with_gemma as gemma_desc
    )
    
    pil_img = page_to_pil(page, dpi=200)
    ocr_text = extract_ocr_text(pil_img)
    
    blocks = parse_ocr_text_to_blocks(ocr_text, page_num)
    
    has_image = detect_meaningful_images(pil_img)
    if use_gemma and has_image:
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()
        prompt = (
            "You are analyzing a scanned page from a technical document. "
            "IGNORE all normal paragraph text. Describe ONLY the diagrams, figures, tables, "
            "photos, or drawings that are visually present on this page. "
            "For each diagram: explain its type, main components, relationships, and any visible labels. "
            "Keep total description under 300 words."
        )
        gemma_description = gemma_desc(img_bytes, prompt=prompt)
        if gemma_description:
            image_block = ImageBlock(description=gemma_description, page=page_num)
            blocks.append(ContentBlock(
                type=BlockType.IMAGE,
                text=gemma_description,
                page=page_num,
                image=image_block,
                metadata={"source": "gemma", "full_page": True}
            ))
    
    return blocks


def build_sections_from_blocks(blocks: List[ContentBlock], total_pages: int) -> List[Section]:
    """
    Build hierarchical sections from flat blocks (same as in scanned_handler).
    """
    sections = []
    section_stack = []
    
    current_section = Section(
        title="Cover",
        level=0,
        section_path=["Cover"],
        page_start=1,
        page_end=1,
        content="",
        tables=[],
        figures=[]
    )
    section_stack.append(current_section)
    
    current_content_parts = []
    current_tables = []
    current_figures = []
    
    for block in blocks:
        if block.type == BlockType.HEADING:
            level = block.level or 2
            if current_section is not None:
                current_section.content = "\n".join(current_content_parts).strip()
                current_section.tables = current_tables[:]
                current_section.figures = current_figures[:]
                current_section.page_end = block.page
                if len(section_stack) > 1:
                    section_stack[-2].children.append(current_section)
                else:
                    sections.append(current_section)
            
            current_section = Section(
                title=block.text,
                level=level,
                section_path=[block.text],
                page_start=block.page,
                page_end=block.page,
                content="",
                tables=[],
                figures=[]
            )
            current_content_parts = []
            current_tables = []
            current_figures = []
            
            while section_stack and section_stack[-1].level >= level:
                section_stack.pop()
            section_stack.append(current_section)
            current_section.section_path = [s.title for s in section_stack]
        
        elif block.type == BlockType.PARAGRAPH:
            current_content_parts.append(block.text)
        elif block.type == BlockType.TABLE and block.table:
            current_tables.append(block.table)
            current_content_parts.append(block.text)
        elif block.type == BlockType.IMAGE and block.image:
            if not any(f.image_bytes == block.image.image_bytes for f in current_figures):
                current_figures.append(block.image)
                desc = block.image.description
                current_content_parts.append(f"[Diagram: {desc}]" if desc else "[Diagram]")
    
    if current_section is not None:
        current_section.content = "\n".join(current_content_parts).strip()
        current_section.tables = current_tables[:]
        current_section.figures = current_figures[:]
        current_section.page_end = total_pages
        if len(section_stack) > 1:
            section_stack[-2].children.append(current_section)
        else:
            sections.append(current_section)
    
    return sections