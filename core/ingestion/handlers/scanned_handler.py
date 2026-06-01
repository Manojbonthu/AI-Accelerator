# core/ingestion/handlers/scanned_handler.py

"""
Scanned PDF Handler – Final tuned version
──────────────────────────────────────────────────────────────────
- PaddleOCR extracts all text from every page.
- YOLO + contour detection finds meaningful visual regions.
- Excludes only very large, text‑heavy tables (>40% page and >80 text lines).
- Keeps diagrams, photos, drawings, and small tables.
- Final: OCR text + optional Gemma description.
"""

import os
import io
import cv2
import fitz
import numpy as np

from PIL import Image
from ultralytics import YOLO
from paddleocr import PaddleOCR
from typing import List, Optional, Tuple

from core.schemas.models import (
    ContentBlock, TableBlock, ImageBlock,
    BlockType, NormalizedDocument, Section
)
from core.ingestion.gemma_client import describe_image_with_gemma


_ocr_engine = None
_yolo_model = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(
            lang='en',
            use_angle_cls=True,
            ocr_version='PP-OCRv4',
            show_log=False
        )
    return _ocr_engine


def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def page_to_pil(page, dpi: int = 200) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data)).convert("RGB")


def extract_ocr_text(pil_image: Image.Image) -> str:
    ocr = get_ocr_engine()
    img_np = np.array(pil_image)
    result = ocr.ocr(img_np)

    text_lines = []
    if result and result[0]:
        for line in result[0]:
            text_lines.append(line[1][0])
    return "\n".join(text_lines)


def detect_meaningful_images(pil_image: Image.Image) -> bool:
    """
    Returns True if the page contains at least one genuine diagram/photo/table.
    Excludes only very large, text‑heavy tables (>40% page and >80 text lines).
    """
    img_w, img_h = pil_image.size
    page_area = img_w * img_h

    # ----- YOLO detection -----
    model = get_yolo_model()
    img_np = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    results = model(img_bgr, conf=0.25, verbose=False)

    yolo_boxes = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    yolo_boxes.append((x1, y1, x2, y2))

    # ----- Contour detection (text masked) -----
    ocr = get_ocr_engine()
    img_np_full = np.array(pil_image)
    result = ocr.ocr(img_np_full)
    text_boxes = []
    if result and result[0]:
        for line in result[0]:
            box = line[0]
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            text_boxes.append((min(xs), min(ys), max(xs), max(ys)))

    gray = cv2.cvtColor(img_np_full, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Mask out text regions
    for (x1, y1, x2, y2) in text_boxes:
        pad = 4
        cv2.rectangle(thresh,
                      (max(0, x1 - pad), max(0, y1 - pad)),
                      (min(img_w, x2 + pad), min(img_h, y2 + pad)),
                      0, -1)

    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=4)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 40 or h < 40:
            continue
        contour_boxes.append((x, y, x + w, y + h))

    # ----- Combine & merge overlapping boxes -----
    all_boxes = yolo_boxes + contour_boxes
    if not all_boxes:
        return False

    # Simple merging (IOU > 0.3)
    merged = list(all_boxes)
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            x1, y1, x2, y2 = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = merged[j]
                ix1 = max(x1, bx1)
                iy1 = max(y1, by1)
                ix2 = min(x2, bx2)
                iy2 = min(y2, by2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a = (x2 - x1) * (y2 - y1)
                area_b = (bx2 - bx1) * (by2 - by1)
                union = area_a + area_b - inter
                iou = inter / union if union > 0 else 0
                if iou > 0.3:
                    x1 = min(x1, bx1)
                    y1 = min(y1, by1)
                    x2 = max(x2, bx2)
                    y2 = max(y2, by2)
                    used[j] = True
                    changed = True
            new_merged.append((x1, y1, x2, y2))
            used[i] = True
        merged = new_merged

    # ----- Filtering -----
    page_text_lines = len(extract_ocr_text(pil_image).splitlines())

    for (x1, y1, x2, y2) in merged:
        w = x2 - x1
        h = y2 - y1
        area = w * h

        # Skip tiny decorative elements
        if area < 20000:
            continue
        # Skip very wide thin bars (headers/footers)
        if w > img_w * 0.7 and h < 150:
            continue
        # Exclude only very large, text‑heavy tables
        if area > 0.4 * page_area and page_text_lines > 80:
            continue

        # Otherwise, it's a meaningful image
        return True

    return False


def parse_ocr_text_to_blocks(text: str, page_num: int) -> List[ContentBlock]:
    blocks = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.isupper() and len(line) < 100:
            blocks.append(ContentBlock(
                type=BlockType.HEADING, text=line, level=2, page=page_num,
                metadata={"source": "paddleocr"}
            ))
        else:
            blocks.append(ContentBlock(
                type=BlockType.PARAGRAPH, text=line, page=page_num,
                metadata={"source": "paddleocr"}
            ))
    return blocks


def build_sections(blocks: List[ContentBlock], total_pages: int) -> List[Section]:
    sections = []
    section_stack = []
    current_section = Section(
        title="Cover", level=0, section_path=["Cover"],
        page_start=1, page_end=1, content="", tables=[], figures=[]
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
                title=block.text, level=level, section_path=[block.text],
                page_start=block.page, page_end=block.page,
                content="", tables=[], figures=[]
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


def extract_scanned(
    file_path: str,
    use_gemma: bool = True,
    save_images_dir: Optional[str] = None
) -> NormalizedDocument:

    doc = fitz.open(file_path)
    file_name = os.path.basename(file_path)
    total_pages = len(doc)
    all_blocks: List[ContentBlock] = []

    pages_with_images = []

    for page_num in range(total_pages):
        page = doc[page_num]
        page_number = page_num + 1
        print(f"\n[Page {page_number}/{total_pages}]")

        pil_img = page_to_pil(page, dpi=200)

        # OCR always
        ocr_text = extract_ocr_text(pil_img)
        print(f"  OCR       : {len(ocr_text.splitlines())} lines")

        # Detect meaningful images (diagrams, photos, tables)
        has_image = detect_meaningful_images(pil_img)

        if has_image:
            pages_with_images.append(page_number)
            print(f"  ✓ Meaningful image(s) detected")

        # Gemma description for pages with images
        gemma_description = ""
        if use_gemma and has_image:
            print(f"  Gemma     : sending full page for diagram description...")
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
            gemma_description = describe_image_with_gemma(img_bytes, prompt=prompt)
            if gemma_description:
                print(f"  Gemma     : description received")
            else:
                print(f"  Gemma     : ⚠ empty response")

        # Add blocks
        text_blocks = parse_ocr_text_to_blocks(ocr_text, page_number)
        all_blocks.extend(text_blocks)

        if gemma_description:
            image_block = ImageBlock(description=gemma_description, page=page_number)
            all_blocks.append(ContentBlock(
                type=BlockType.IMAGE,
                text=gemma_description,
                page=page_number,
                image=image_block,
                metadata={"source": "gemma", "full_page": True}
            ))

        all_blocks.append(ContentBlock(
            type=BlockType.PAGE_BREAK,
            text=f"--- Page {page_number} ---",
            page=page_number,
            metadata={"document_name": file_name}
        ))

    doc.close()

    print("\n" + "─" * 50)
    print("FINAL SUMMARY")
    print("─" * 50)
    print(f"  Total pages        : {total_pages}")
    print(f"  Pages with images  : {len(pages_with_images)}")
    if pages_with_images:
        print(f"  Pages: {pages_with_images}")
    print(f"  Gemma enabled      : {use_gemma}")
    print("─" * 50)

    sections = build_sections(all_blocks, total_pages)

    return NormalizedDocument(
        file_name=file_name,
        total_pages=total_pages,
        pdf_type="scanned",
        sections=sections,
        blocks=all_blocks
    )


# ============================================================
# NEW FUNCTION for mixed_handler – extracts a single scanned page
# ============================================================
def extract_scanned_page_content(page: fitz.Page, page_num: int, use_gemma: bool = True) -> List[ContentBlock]:
    """
    Extract content from a single scanned page using OCR + YOLO + optional Gemma.
    Returns a list of ContentBlock.
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