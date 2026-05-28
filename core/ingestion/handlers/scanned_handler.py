"""
core/ingestion/handlers/scanned_handler.py

Simplified handler for scanned PDFs using only PaddleOCR + embedded image extraction.
- No Unstructured, no torch – avoids Windows DLL issues.
- Extracts text, groups into paragraphs, detects headings.
- Extracts embedded images (logos, diagrams) and describes them via Gemma.
- Builds hierarchical sections identical to digital handler.
"""

import os
import io
import hashlib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image
import fitz  # pymupdf
from paddleocr import PaddleOCR

from core.schemas.models import (
    ContentBlock, TableBlock, ImageBlock, BlockType,
    NormalizedDocument, Section
)
from core.ingestion.gemma_client import describe_image_with_gemma

# ------------------------------------------------------------------
# Global OCR engine and image registry
# ------------------------------------------------------------------
_ocr_engine = None
_image_registry: Dict[str, ImageBlock] = {}

def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(lang='en', use_angle_cls=True, enable_mkldnn=False)
    return _ocr_engine


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


# ------------------------------------------------------------------
# Helper: PDF page to PIL image
# ------------------------------------------------------------------
def page_to_pil(page, dpi=200) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data))


# ------------------------------------------------------------------
# Run PaddleOCR and return list of text blocks with bounding boxes
# ------------------------------------------------------------------
def run_ocr(img: Image.Image) -> List[Dict]:
    ocr = get_ocr_engine()
    img_np = np.array(img)
    result = ocr.ocr(img_np)
    blocks = []
    if result and result[0]:
        for line in result[0]:
            bbox = line[0]
            text = line[1][0]
            conf = line[1][1]
            x0 = min(p[0] for p in bbox)
            y0 = min(p[1] for p in bbox)
            x1 = max(p[0] for p in bbox)
            y1 = max(p[1] for p in bbox)
            blocks.append({
                "bbox": (x0, y0, x1, y1),
                "text": text,
                "confidence": conf
            })
    return blocks


# ------------------------------------------------------------------
# Group OCR blocks into paragraphs (by vertical gaps)
# ------------------------------------------------------------------
def group_into_paragraphs(blocks: List[Dict]) -> List[Dict]:
    if not blocks:
        return []
    blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    paragraphs = []
    current = []
    last_y = None
    for b in blocks:
        y0 = b["bbox"][1]
        y1 = b["bbox"][3]
        height = y1 - y0
        if last_y is None:
            current.append(b)
            last_y = y0
        else:
            gap = y0 - last_y
            if gap < height * 1.5:
                current.append(b)
            else:
                if current:
                    paragraphs.append({
                        "blocks": current,
                        "text": " ".join([p["text"] for p in current]),
                        "bbox": (
                            min(p["bbox"][0] for p in current),
                            min(p["bbox"][1] for p in current),
                            max(p["bbox"][2] for p in current),
                            max(p["bbox"][3] for p in current)
                        ),
                        "avg_height": np.mean([p["bbox"][3] - p["bbox"][1] for p in current])
                    })
                current = [b]
                last_y = y0
    if current:
        paragraphs.append({
            "blocks": current,
            "text": " ".join([p["text"] for p in current]),
            "bbox": (min(p["bbox"][0] for p in current),
                     min(p["bbox"][1] for p in current),
                     max(p["bbox"][2] for p in current),
                     max(p["bbox"][3] for p in current)),
            "avg_height": np.mean([p["bbox"][3] - p["bbox"][1] for p in current])
        })
    return paragraphs


# ------------------------------------------------------------------
# Classify paragraph as heading or normal text
# ------------------------------------------------------------------
def classify_paragraph(para: Dict) -> str:
    text = para["text"].strip()
    avg_height = para["avg_height"]
    # Heading heuristics: larger font OR short all-caps OR ends with colon OR starts with number
    if avg_height > 25:
        return "heading"
    if len(text) < 100 and (text.isupper() or text.endswith(':') or (text[0].isdigit() and '.' in text[:5])):
        return "heading"
    return "text"


# ------------------------------------------------------------------
# Convert paragraphs to ContentBlocks
# ------------------------------------------------------------------
def paragraphs_to_blocks(paragraphs: List[Dict], page_num: int, doc_name: str) -> List[ContentBlock]:
    blocks = []
    for para in paragraphs:
        if classify_paragraph(para) == "heading":
            blocks.append(ContentBlock(
                type=BlockType.HEADING,
                text=para["text"],
                level=2,  # could be refined by avg_height
                page=page_num,
                metadata={"source": "ocr", "avg_font_height": para["avg_height"]}
            ))
        else:
            blocks.append(ContentBlock(
                type=BlockType.PARAGRAPH,
                text=para["text"],
                page=page_num,
                metadata={"source": "ocr", "avg_font_height": para["avg_height"]}
            ))
    return blocks


# ------------------------------------------------------------------
# Extract embedded raster images from a PDF page
# ------------------------------------------------------------------
def extract_embedded_images(page, page_num: int, doc: fitz.Document, use_gemma: bool) -> List[ContentBlock]:
    image_blocks = []
    img_list = page.get_images(full=True)
    for img in img_list:
        xref = img[0]
        try:
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            # Get image rectangle(s) – usually one
            rects = page.get_image_rects(img)
            bbox = rects[0] if rects else (0, 0, 0, 0)
            img_hash, img_block = _get_or_create_image(image_bytes, page_num, use_gemma)
            image_blocks.append(ContentBlock(
                type=BlockType.IMAGE,
                text=img_block.description or "",
                page=page_num,
                image=img_block,
                metadata={"document_name": doc.name, "bbox": bbox}
            ))
        except Exception:
            continue
    return image_blocks


# ------------------------------------------------------------------
# Build section tree (identical to digital handler)
# ------------------------------------------------------------------
def build_sections(blocks: List[ContentBlock], total_pages: int) -> List[Section]:
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
            heading_text = block.text

            # Finalize previous section
            if current_section is not None:
                current_section.content = "\n".join(current_content_parts).strip()
                current_section.tables = current_tables[:]
                current_section.figures = current_figures[:]
                current_section.page_end = block.page
                if len(section_stack) > 1:
                    section_stack[-2].children.append(current_section)
                else:
                    sections.append(current_section)

            # Start new section
            current_section = Section(
                title=heading_text,
                level=level,
                section_path=[heading_text],
                page_start=block.page,
                page_end=block.page,
                content="",
                tables=[],
                figures=[]
            )
            current_content_parts = []
            current_tables = []
            current_figures = []

            while len(section_stack) > 0 and section_stack[-1].level >= level:
                section_stack.pop()
            section_stack.append(current_section)
            current_section.section_path = [s.title for s in section_stack]
        else:
            if block.type == BlockType.PARAGRAPH:
                current_content_parts.append(block.text)
            elif block.type == BlockType.TABLE and block.table:
                current_tables.append(block.table)
                current_content_parts.append(block.text)
            elif block.type == BlockType.IMAGE and block.image:
                # Deduplicate within section (by bytes)
                if not any(f.image_bytes == block.image.image_bytes for f in current_figures):
                    current_figures.append(block.image)
                    if block.image.description:
                        current_content_parts.append(f"[Image: {block.image.description}]")
                    elif block.image.caption:
                        current_content_parts.append(f"[Image: {block.image.caption}]")
                    else:
                        current_content_parts.append("[Image]")

    # Finalize last section
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


# ------------------------------------------------------------------
# Main extraction function
# ------------------------------------------------------------------
def extract_scanned(file_path: str, use_gemma: bool = True) -> NormalizedDocument:
    # Reset image registry for this document
    global _image_registry
    _image_registry = {}

    doc = fitz.open(file_path)
    file_name = os.path.basename(file_path)
    total_pages = len(doc)

    all_blocks = []

    for page_num in range(total_pages):
        page = doc[page_num]
        page_number = page_num + 1

        # Render page to image for OCR
        pil_img = page_to_pil(page)

        # 1. OCR and paragraph grouping
        ocr_blocks = run_ocr(pil_img)
        paragraphs = group_into_paragraphs(ocr_blocks)
        text_blocks = paragraphs_to_blocks(paragraphs, page_number, file_name)

        # 2. Extract embedded images
        image_blocks = extract_embedded_images(page, page_number, doc, use_gemma)

        # 3. Merge text and image blocks (preserve reading order? images are separate)
        # For simplicity, add all text blocks first, then images after (or interlace by y-coordinate if needed)
        # To keep order, we can sort all blocks by their bounding box y-position (for images, use bbox)
        # But images from get_images have bbox, so we can combine and sort.
        combined = text_blocks + image_blocks
        # Sort by page, then by bbox y (images have bbox)
        combined.sort(key=lambda b: (b.page, b.metadata.get("bbox", (0,0,0,0))[1] if b.type == BlockType.IMAGE else 0))
        all_blocks.extend(combined)

        # Add page break marker
        all_blocks.append(ContentBlock(
            type=BlockType.PAGE_BREAK,
            text=f"--- Page {page_number} ---",
            page=page_number,
            metadata={"document_name": file_name}
        ))

    doc.close()

    # Build section hierarchy
    sections = build_sections(all_blocks, total_pages)

    return NormalizedDocument(
        file_name=file_name,
        total_pages=total_pages,
        pdf_type="scanned",
        sections=sections,
        blocks=all_blocks
    )