"""
core/ingestion/handlers/digital_handler.py

Extracts content from digital PDFs using pymupdf.
Now also sends embedded images to Gemma for description.
"""

import fitz  # pymupdf
import os
from typing import List, Optional
from core.schemas.models import (
    ContentBlock, TableBlock, ImageBlock,
    BlockType, NormalizedDocument
)
from core.ingestion.gemma_client import describe_image_with_gemma


def extract_digital(file_path: str, use_gemma: bool = True) -> NormalizedDocument:
    doc = fitz.open(file_path)
    file_name = os.path.basename(file_path)
    total_pages = len(doc)

    all_blocks = []
    section_stack = []
    section_counter = 0

    for page_num in range(total_pages):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        all_blocks.append(ContentBlock(
            type=BlockType.PAGE_BREAK,
            text=f"--- Page {page_num + 1} ---",
            page=page_num + 1
        ))

        for block in blocks:
            if block["type"] == 0:  # Text
                text_blocks = process_text_block(block, page_num + 1, section_counter)
                for tb in text_blocks:
                    if tb.type == BlockType.HEADING:
                        section_counter += 1
                        tb.section_id = f"section_{section_counter}"
                        if tb.level:
                            while section_stack and section_stack[-1][0] >= tb.level:
                                section_stack.pop()
                            section_stack.append((tb.level, tb.text, tb.section_id))
                    if section_stack:
                        tb.section_id = section_stack[-1][2]
                    all_blocks.append(tb)

            elif block["type"] == 1:  # Image
                image_blocks = process_image_block(block, doc, page_num + 1, use_gemma)
                all_blocks.extend(image_blocks)

    doc.close()

    return NormalizedDocument(
        file_name=file_name,
        total_pages=total_pages,
        pdf_type="digital",
        blocks=all_blocks
    )


def process_text_block(block: dict, page_num: int, section_counter: int) -> List[ContentBlock]:
    results = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            text = span["text"].strip()
            if not text:
                continue
            font_size = span["size"]
            is_bold = "Bold" in span.get("font", "")
            if font_size >= 16 or (font_size >= 14 and is_bold):
                level = 1 if font_size >= 18 else 2 if font_size >= 16 else 3
                results.append(ContentBlock(
                    type=BlockType.HEADING, text=text, level=level, page=page_num,
                    metadata={"font_size": font_size, "is_bold": is_bold}
                ))
            else:
                if text.lower().startswith(("fig", "figure", "table", "image")) and len(text) < 150:
                    results.append(ContentBlock(type=BlockType.CAPTION, text=text, page=page_num))
                else:
                    results.append(ContentBlock(type=BlockType.PARAGRAPH, text=text, page=page_num))
    return results


def process_image_block(block: dict, doc: fitz.Document, page_num: int, use_gemma: bool = True) -> List[ContentBlock]:
    """
    Extract image bytes from the PDF block.
    1. Try xref extraction (gives original format).
    2. Fall back to raw block image.
    If use_gemma is True, sends image to Gemma for description.
    """
    results = []
    # 1) Try xref from block["images"]
    for img_info in block.get("images", []):
        try:
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            mime = f"image/{base_image['ext']}"
            image_block = ImageBlock(image_bytes=image_bytes, mime_type=mime, page=page_num)
            if use_gemma:
                desc = describe_image_with_gemma(image_bytes)
                if desc:
                    image_block.description = desc
            results.append(ContentBlock(
                type=BlockType.IMAGE,
                text=image_block.description or "",
                page=page_num,
                image=image_block
            ))
        except Exception:
            pass

    # 2) Fallback to raw block["image"]
    if not results and block.get("image"):
        try:
            image_bytes = block["image"]
            image_block = ImageBlock(image_bytes=image_bytes, mime_type="image/png", page=page_num)
            if use_gemma:
                desc = describe_image_with_gemma(image_bytes)
                if desc:
                    image_block.description = desc
            results.append(ContentBlock(
                type=BlockType.IMAGE,
                text=image_block.description or "",
                page=page_num,
                image=image_block
            ))
        except Exception:
            pass

    return results


def extract_tables_from_page(doc: fitz.Document, page_num: int) -> List[ContentBlock]:
    page = doc[page_num]
    tables = page.find_tables()
    table_blocks = []
    for table in tables.tables:
        if table.header and table.extract():
            headers = [str(h) for h in table.header.names]
            rows = [[str(cell) for cell in row] for row in table.extract()]
            table_blocks.append(ContentBlock(
                type=BlockType.TABLE,
                text=table_to_markdown(headers, rows),
                page=page_num + 1,
                table=TableBlock(headers=headers, rows=rows, page=page_num + 1)
            ))
    return table_blocks


def table_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---" for _ in headers]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)