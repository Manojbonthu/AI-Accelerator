"""
core/ingestion/handlers/digital_handler.py

Extracts content from digital PDFs and builds a hierarchical section tree.
Removes noise, deduplicates images, and prepares for section‑based chunking.
Now includes comprehensive per‑page metrics for production‑grade PDF analysis.
Also intelligently sends only truly valuable diagrams (vector‑rich pages) to Gemma.
"""

import fitz
import os
import re
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from core.schemas.models import (
    ContentBlock, TableBlock, ImageBlock, BlockType,
    NormalizedDocument, Section
)
from core.ingestion.gemma_client import describe_image_with_gemma


# ------------------------------------------------------------------
# Configuration for Gemma decisions
# ------------------------------------------------------------------
VECTOR_THRESHOLD = 20   # pages with more vector drawings are considered diagram candidates

SKIP_SECTION_KEYWORDS = [
    "indicator lights",
    "symbols",
    "compliance",
    "warranty",
    "safety information",
    "notes",
    "troubleshooting"
]


def _should_send_page_to_gemma(vector_count: int, section_title: str) -> bool:
    """
    Decide whether to send a page to Gemma for description.
    """
    if vector_count <= VECTOR_THRESHOLD:
        return False
    title_lower = section_title.lower()
    for keyword in SKIP_SECTION_KEYWORDS:
        if keyword in title_lower:
            return False
    return True


def extract_digital(file_path: str, use_gemma: bool = True) -> NormalizedDocument:
    """
    Extract a digital PDF and return a NormalizedDocument with a section hierarchy.
    """
    # Reset image registry for this document (prevents cross‑document leaks)
    global _image_registry
    _image_registry = {}

    doc = fitz.open(file_path)
    file_name = os.path.basename(file_path)
    total_pages = len(doc)

    # Final top‑level sections
    sections: List[Section] = []
    # Stack to track current hierarchy path
    section_stack: List[Section] = []

    # Create a default "Cover" section to capture content before the first heading
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

    current_content_parts: List[str] = []
    current_tables: List[TableBlock] = []
    current_figures: List[ImageBlock] = []

    for page_num in range(total_pages):
        page = doc[page_num]
        page_number = page_num + 1

        # ----- Compute per‑page production metrics -----
        text = page.get_text().strip()
        text_length = len(text)
        has_text = text_length > 0

        raster_images = len(page.get_images(full=True))
        vector_drawings = len(page.get_drawings())
        annotations = len(list(page.annots())) if hasattr(page, 'annots') else 0
        links = len(page.get_links())

        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        image_area = 0.0
        if raster_images > 0:
            img_list = page.get_images(full=True)
            for img in img_list:
                rects = page.get_image_rects(img)
                for r in rects:
                    image_area += r.width * r.height
        image_coverage_ratio = image_area / page_area if page_area > 0 else 0.0

        # ----- Extract text, images, tables -----
        text_blocks = _extract_text_blocks_with_bbox(page, page_number, file_name)
        image_blocks = _extract_image_blocks_with_bbox(page, doc, page_number, use_gemma)
        table_blocks = _extract_table_blocks_with_bbox(page, page_number, file_name)

        # Merge and sort
        all_elements = text_blocks + image_blocks + table_blocks
        all_elements.sort(key=lambda x: x['bbox'][1])
        all_elements = _merge_adjacent_headings(all_elements)

        # ----- Add page break block with metadata -----
        page_break_metadata = {
            "document_name": file_name,
            "has_text": has_text,
            "text_length": text_length,
            "raster_images": raster_images,
            "vector_drawings": vector_drawings,
            "annotations": annotations,
            "links": links,
            "image_coverage_ratio": round(image_coverage_ratio, 4)
        }
        page_break_block = ContentBlock(
            type=BlockType.PAGE_BREAK,
            text=f"--- Page {page_number} ---",
            page=page_number,
            metadata=page_break_metadata
        )
        current_content_parts.append(page_break_block.text)

        # ----- Decide whether to send the page (as an image) to Gemma -----
        current_section_title = section_stack[-1].title if section_stack else "Cover"
        send_to_gemma = (use_gemma and
                         _should_send_page_to_gemma(vector_drawings, current_section_title))
        if send_to_gemma:
            # Render the whole page at 2x resolution for better clarity
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            description = describe_image_with_gemma(img_bytes)
            if description:
                current_content_parts.append(f"[Full-page diagram description from Gemini: {description}]")
                current_figures.append(ImageBlock(description=description, page=page_number))

        # ----- Process all elements -----
        for elem in all_elements:
            block_type = elem['type']
            text = elem.get('text', '')
            bbox = elem['bbox']

            if block_type == BlockType.PARAGRAPH and _is_noise(text):
                continue
            if block_type == BlockType.PARAGRAPH and _looks_like_table_row(text):
                continue

            if block_type == BlockType.HEADING:
                level = elem.get('level', 2)
                heading_text = _clean_heading(text)

                # Finalize current section
                if current_section is not None:
                    current_section.content = "\n".join(current_content_parts).strip()
                    current_section.tables = current_tables[:]
                    current_section.figures = current_figures[:]
                    current_section.page_end = page_number
                    if len(section_stack) > 1:
                        section_stack[-2].children.append(current_section)
                    else:
                        sections.append(current_section)

                # Start new section
                current_section = Section(
                    title=heading_text,
                    level=level,
                    section_path=[heading_text],
                    page_start=page_number,
                    page_end=page_number,
                    content="",
                    tables=[],
                    figures=[]
                )
                current_content_parts = []
                current_tables = []
                current_figures = []

                # Update section stack
                while len(section_stack) > 0 and section_stack[-1].level >= level:
                    section_stack.pop()
                section_stack.append(current_section)
                current_section.section_path = [s.title for s in section_stack]
                continue

            # Non‑heading content
            if block_type == BlockType.PARAGRAPH:
                current_content_parts.append(text)
            elif block_type == BlockType.TABLE and 'table_block' in elem:
                tbl = elem['table_block']
                current_tables.append(tbl)
                md = _table_to_markdown(tbl.headers, tbl.rows)
                current_content_parts.append(md)
            elif block_type == BlockType.IMAGE and 'image_block' in elem:
                img = elem['image_block']
                if not any(f.image_bytes == img.image_bytes for f in current_figures):
                    current_figures.append(img)
                    if img.description:
                        current_content_parts.append(f"[Image: {img.description}]")
                    elif img.caption:
                        current_content_parts.append(f"[Image: {img.caption}]")
                    else:
                        current_content_parts.append("[Image]")

    # Finalize the last section
    if current_section is not None:
        current_section.content = "\n".join(current_content_parts).strip()
        current_section.tables = current_tables[:]
        current_section.figures = current_figures[:]
        current_section.page_end = total_pages
        if len(section_stack) > 1:
            section_stack[-2].children.append(current_section)
        else:
            sections.append(current_section)

    doc.close()

    # Post-process: remove page headers/footers from section content
    doc_title = os.path.splitext(file_name)[0]
    if sections and sections[0].children:
        first_heading = sections[0].children[0].title
        if first_heading and len(first_heading) > 3:
            doc_title = first_heading
    _remove_page_headers_from_sections(sections, doc_title)

    return NormalizedDocument(
        file_name=file_name,
        total_pages=total_pages,
        pdf_type="digital",
        sections=sections,
        blocks=[]
    )


# ------------------------------------------------------------------
# Global image registry (reset per call inside extract_digital)
# ------------------------------------------------------------------
_image_registry: Dict[str, ImageBlock] = {}


def _get_or_create_image(image_bytes: bytes, page_num: int, use_gemma: bool) -> Tuple[str, ImageBlock]:
    img_hash = hashlib.sha256(image_bytes).hexdigest()
    if img_hash in _image_registry:
        return img_hash, _image_registry[img_hash]
    image_block = ImageBlock(
        image_bytes=image_bytes,
        mime_type="image/png",
        page=page_num,
        confidence=1.0
    )
    if use_gemma:
        desc = describe_image_with_gemma(image_bytes)
        if desc:
            image_block.description = desc
    _image_registry[img_hash] = image_block
    return img_hash, image_block


def _is_noise(text: str) -> bool:
    if re.search(r"(page|pg\.?)\s*\d+\s+of\s+\d+", text, re.IGNORECASE):
        return True
    if re.match(r"^\s*\d+\s*$", text):
        return True
    return False


def _looks_like_table_row(text: str) -> bool:
    if '|' in text and '---' in text:
        return True
    if re.search(r'\S+\s{3,}\S+', text) and len(text.split()) >= 4:
        return True
    lines = text.split('\n')
    kv_count = 0
    for line in lines:
        if re.match(r'^\s*\w+\s*:\s*\S', line):
            kv_count += 1
    if kv_count >= 2:
        return True
    field_pattern = r'\b(Sample Date|Prepared by|Created and Tested Using|Features Demonstrated)\b'
    if re.search(field_pattern, text, re.IGNORECASE):
        return True
    return False


def _clean_heading(text: str) -> str:
    text = " ".join(text.split())
    text = re.sub(r'(?<=[A-Z]) (?=[A-Z][A-Za-z]+)', '', text)
    text = re.sub(r'(?<=[A-Z]) (?=[A-Z])(?! )', '', text)
    return text


def _merge_adjacent_headings(elements: List[Dict]) -> List[Dict]:
    if not elements:
        return elements
    merged = []
    i = 0
    while i < len(elements):
        elem = elements[i]
        if elem.get("type") != BlockType.HEADING:
            merged.append(elem)
            i += 1
            continue
        group = [elem]
        j = i + 1
        while j < len(elements) and elements[j].get("type") == BlockType.HEADING and elements[j].get("page") == elem.get("page"):
            group.append(elements[j])
            j += 1
        if len(group) > 1:
            combined_text = " ".join([g["text"] for g in group])
            combined_text = _clean_heading(combined_text)
            best_level = min(g.get("level", 2) for g in group)
            merged_elem = {
                "type": BlockType.HEADING,
                "text": combined_text,
                "level": best_level,
                "page": elem["page"],
                "bbox": group[0]["bbox"]
            }
            merged.append(merged_elem)
            i = j
        else:
            elem["text"] = _clean_heading(elem["text"])
            merged.append(elem)
            i += 1
    return merged


def _remove_page_headers_from_sections(sections: List[Section], doc_title: str):
    for section in sections:
        if section.content:
            lines = section.content.split("\n")
            cleaned_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.lower() == doc_title.lower():
                    continue
                if re.search(r"(page|pg\.?)\s*\d+\s+of\s+\d+", stripped, re.IGNORECASE):
                    continue
                if re.match(r"^\s*\d+\s*$", stripped):
                    continue
                if stripped.lower() == "pdf bookmark sample":
                    continue
                cleaned_lines.append(line)
            section.content = "\n".join(cleaned_lines)
        _remove_page_headers_from_sections(section.children, doc_title)


def _extract_text_blocks_with_bbox(page, page_num: int, doc_name: str) -> List[Dict]:
    blocks_data = page.get_text("dict")["blocks"]
    lines_dict: Dict[float, List[Dict]] = {}
    for block in blocks_data:
        if block["type"] != 0:
            continue
        for line in block.get("lines", []):
            y_top = line["bbox"][1]
            spans = []
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not text:
                    continue
                spans.append({
                    "text": text,
                    "size": span["size"],
                    "font": span.get("font", ""),
                    "bbox": span["bbox"]
                })
            if spans:
                if y_top not in lines_dict:
                    lines_dict[y_top] = []
                lines_dict[y_top].extend(spans)
    sorted_y = sorted(lines_dict.keys())
    text_lines = []
    for y_top in sorted_y:
        spans = lines_dict[y_top]
        line_text = " ".join([s["text"] for s in spans])
        first_span = spans[0]
        text_lines.append({
            "text": line_text,
            "size": first_span["size"],
            "font": first_span["font"],
            "bbox": first_span["bbox"],
            "page": page_num
        })
    results = []
    for line in text_lines:
        text = line["text"]
        font_size = line["size"]
        font = line["font"]
        is_bold = "Bold" in font or "bold" in font.lower()
        bbox = line["bbox"]
        if font_size >= 14 or (font_size >= 12 and is_bold):
            level = 1 if font_size >= 18 else 2 if font_size >= 14 else 3
            results.append({
                "type": BlockType.HEADING,
                "text": text,
                "level": level,
                "page": page_num,
                "bbox": bbox
            })
        else:
            if text.lower().startswith(("fig", "figure", "table", "image", "photo")) and len(text) < 200:
                results.append({
                    "type": BlockType.CAPTION,
                    "text": text,
                    "page": page_num,
                    "bbox": bbox
                })
            else:
                results.append({
                    "type": BlockType.PARAGRAPH,
                    "text": text,
                    "page": page_num,
                    "bbox": bbox
                })
    return results


def _extract_image_blocks_with_bbox(page, doc, page_num: int, use_gemma: bool) -> List[Dict]:
    results = []
    image_list = page.get_images(full=True)
    for img in image_list:
        xref = img[0]
        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            rects = page.get_image_rects(img)
            bbox = rects[0] if rects else (0, 0, 0, 0)
            img_hash, img_block = _get_or_create_image(image_bytes, page_num, use_gemma)
            results.append({
                "type": BlockType.IMAGE,
                "text": img_block.description or "",
                "image_block": img_block,
                "page": page_num,
                "bbox": bbox
            })
        except Exception:
            continue
    return results


def _extract_table_blocks_with_bbox(page, page_num: int, doc_name: str) -> List[Dict]:
    results = []
    tables = page.find_tables()
    for table in tables.tables:
        if not table.header or not table.extract():
            continue
        headers = [str(h) for h in table.header.names]
        rows = [[str(cell) for cell in row] for row in table.extract()]
        table_block = TableBlock(headers=headers, rows=rows, page=page_num)
        bbox = table.bbox if hasattr(table, 'bbox') else (0, 0, 0, 0)
        results.append({
            "type": BlockType.TABLE,
            "text": _table_to_markdown(headers, rows),
            "table_block": table_block,
            "page": page_num,
            "bbox": bbox
        })
    return results


def _table_to_markdown(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---" for _ in headers]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)