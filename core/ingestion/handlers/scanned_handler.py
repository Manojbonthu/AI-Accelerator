# core/ingestion/handlers/scanned_handler.py

"""
Scanned PDF Handler – Production Pipeline with Gemma enabled by default.

Flow:
1. PaddleOCR → extract all text
2. YOLO + OpenCV contours → detect visual/diagram regions
3. Filter regions by size, aspect ratio
4. **If use_gemma=True (default)**, send each region to Gemma for description
5. Combine OCR text + diagram descriptions
6. Build hierarchical sections
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


# ============================================================
# GLOBAL MODELS (loaded once)
# ============================================================

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


# ============================================================
# PDF PAGE → PIL
# ============================================================

def page_to_pil(page, dpi: int = 200) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi)
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data)).convert("RGB")


# ============================================================
# STEP 1: OCR EXTRACTION
# ============================================================

def extract_ocr_text(
    pil_image: Image.Image
) -> Tuple[str, List[Tuple[int, int, int, int]]]:
    ocr = get_ocr_engine()
    img_np = np.array(pil_image)
    result = ocr.ocr(img_np)

    text_lines = []
    text_boxes = []

    if result and result[0]:
        for line in result[0]:
            box  = line[0]
            text = line[1][0]
            text_lines.append(text)

            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            text_boxes.append((min(xs), min(ys), max(xs), max(ys)))

    return "\n".join(text_lines), text_boxes


# ============================================================
# STEP 2A: YOLO OBJECT DETECTION
# ============================================================

def detect_yolo_regions(
    pil_image: Image.Image,
    confidence: float = 0.25
) -> List[Tuple[int, int, int, int]]:
    model   = get_yolo_model()
    img_np  = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    results = model(img_bgr, conf=confidence, verbose=False)

    boxes = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                boxes.append((x1, y1, x2, y2))

    return boxes


# ============================================================
# STEP 2B: CONTOUR-BASED VISUAL REGION DETECTION
# ============================================================

def detect_contour_regions(
    pil_image: Image.Image,
    text_boxes: List[Tuple[int, int, int, int]],
    min_area: int  = 5000,
    min_side: int  = 40,
    aspect_ratio_max: float = 10.0
) -> List[Tuple[int, int, int, int]]:
    img_np       = np.array(pil_image)
    img_h, img_w = img_np.shape[:2]

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Mask out text regions
    for (x1, y1, x2, y2) in text_boxes:
        pad = 4
        cv2.rectangle(
            thresh,
            (max(0, x1 - pad), max(0, y1 - pad)),
            (min(img_w, x2 + pad), min(img_h, y2 + pad)),
            0, -1
        )

    kernel  = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=4)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area   = w * h
        aspect = max(w, h) / (min(w, h) + 1e-5)

        if area   < min_area:        continue
        if w      < min_side:        continue
        if h      < min_side:        continue
        if aspect > aspect_ratio_max: continue

        regions.append((x, y, x + w, y + h))

    return regions


# ============================================================
# FILTER: KEEP ONLY REAL IMAGE REGIONS
# ============================================================

def filter_real_image_regions(
    regions: List[Tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
    min_width: int  = 80,
    min_height: int = 100,
    min_area: int   = 10000,
    page_fraction_max: float = 0.30
) -> List[Tuple[int, int, int, int]]:
    page_area    = img_w * img_h
    real_regions = []

    for (x1, y1, x2, y2) in regions:
        w    = x2 - x1
        h    = y2 - y1
        area = w * h

        if w < min_width or h < min_height:
            continue

        if area < min_area:
            continue

        # Skip near-full-page regions (borders, backgrounds)
        if area > page_fraction_max * page_area:
            continue

        # Skip wide thin header/footer bars
        if w > img_w * 0.7 and h < 200:
            continue

        # Skip extreme aspect ratios (thin lines, tall slivers)
        aspect = w / (h + 1e-5)
        if aspect > 8 or aspect < 0.15:
            continue

        real_regions.append((x1, y1, x2, y2))

    return real_regions


# ============================================================
# MERGE OVERLAPPING BOXES
# ============================================================

def merge_boxes(
    boxes: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.05
) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []

    merged  = list(boxes)
    changed = True

    while changed:
        changed = False
        result  = []
        used    = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]:
                continue

            x1, y1, x2, y2 = merged[i]

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue

                bx1, by1, bx2, by2 = merged[j]

                ix1 = max(x1, bx1);  iy1 = max(y1, by1)
                ix2 = min(x2, bx2);  iy2 = min(y2, by2)

                inter  = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a = (x2 - x1) * (y2 - y1)
                area_b = (bx2 - bx1) * (by2 - by1)
                union  = area_a + area_b - inter
                iou    = inter / union if union > 0 else 0

                if iou > iou_threshold:
                    x1 = min(x1, bx1);  y1 = min(y1, by1)
                    x2 = max(x2, bx2);  y2 = max(y2, by2)
                    used[j] = True
                    changed  = True

            result.append((x1, y1, x2, y2))
            used[i] = True

        merged = result

    return merged


# ============================================================
# STEP 3: CROP REGIONS → GEMMA
# ============================================================

def analyze_regions_with_gemma(
    pil_image: Image.Image,
    regions:   List[Tuple[int, int, int, int]]
) -> List[dict]:
    explanations = []
    img_w, img_h = pil_image.size

    for idx, (x1, y1, x2, y2) in enumerate(regions):
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(img_w, x2);  y2 = min(img_h, y2)

        if (x2 - x1) < 30 or (y2 - y1) < 30:
            continue

        cropped = pil_image.crop((x1, y1, x2, y2))
        buf     = io.BytesIO()
        cropped.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        prompt = (
            "You are analyzing a cropped region from a technical document. "
            "This region contains a diagram, figure, chart, or technical drawing. "
            "IGNORE any surrounding page text. "
            "Describe ONLY what is visually present in this image:\n"
            "- Type of diagram/figure (flowchart, bar chart, photo, engineering drawing, etc.)\n"
            "- Main components or parts\n"
            "- Relationships between components\n"
            "- Any visible labels that are part of the diagram itself\n"
            "Keep description under 150 words. Do not repeat page text."
        )

        response = describe_image_with_gemma(img_bytes, prompt=prompt)

        if response and response.strip():
            explanations.append({
                "region":      [x1, y1, x2, y2],
                "description": response.strip()
            })
            print(f"  Region {idx + 1}: Gemma described successfully")
        else:
            print(f"  Region {idx + 1}: Gemma returned empty")

    return explanations


# ============================================================
# OCR TEXT → CONTENT BLOCKS
# ============================================================

def parse_ocr_text_to_blocks(
    text:     str,
    page_num: int
) -> List[ContentBlock]:
    blocks = []

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.isupper() and len(line) < 100:
            blocks.append(ContentBlock(
                type=BlockType.HEADING,
                text=line,
                level=2,
                page=page_num,
                metadata={"source": "paddleocr"}
            ))
        else:
            blocks.append(ContentBlock(
                type=BlockType.PARAGRAPH,
                text=line,
                page=page_num,
                metadata={"source": "paddleocr"}
            ))

    return blocks


# ============================================================
# BUILD SECTION TREE
# ============================================================

def build_sections(
    blocks:      List[ContentBlock],
    total_pages: int
) -> List[Section]:
    sections      = []
    section_stack = []

    current_section = Section(
        title="Cover", level=0,
        section_path=["Cover"],
        page_start=1, page_end=1,
        content="", tables=[], figures=[]
    )
    section_stack.append(current_section)

    current_content_parts = []
    current_tables        = []
    current_figures       = []

    for block in blocks:

        if block.type == BlockType.HEADING:
            level = block.level or 2

            if current_section is not None:
                current_section.content = "\n".join(current_content_parts).strip()
                current_section.tables  = current_tables[:]
                current_section.figures = current_figures[:]
                current_section.page_end = block.page

                if len(section_stack) > 1:
                    section_stack[-2].children.append(current_section)
                else:
                    sections.append(current_section)

            current_section = Section(
                title=block.text, level=level,
                section_path=[block.text],
                page_start=block.page, page_end=block.page,
                content="", tables=[], figures=[]
            )
            current_content_parts = []
            current_tables        = []
            current_figures       = []

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
            if not any(
                f.image_bytes == block.image.image_bytes
                for f in current_figures
            ):
                current_figures.append(block.image)
                desc = block.image.description
                current_content_parts.append(
                    f"[Diagram: {desc}]" if desc else "[Diagram]"
                )

    # Finalize last section
    if current_section is not None:
        current_section.content  = "\n".join(current_content_parts).strip()
        current_section.tables   = current_tables[:]
        current_section.figures  = current_figures[:]
        current_section.page_end = total_pages

        if len(section_stack) > 1:
            section_stack[-2].children.append(current_section)
        else:
            sections.append(current_section)

    return sections


# ============================================================
# MAIN EXTRACTION FUNCTION
# ============================================================

def extract_scanned(
    file_path:      str,
    use_gemma:      bool = True,        # ← Changed: default True
    save_images_dir: Optional[str] = None
) -> NormalizedDocument:

    doc        = fitz.open(file_path)
    file_name  = os.path.basename(file_path)
    total_pages = len(doc)
    all_blocks: List[ContentBlock] = []

    page_image_counts = {}

    for page_num in range(total_pages):
        page        = doc[page_num]
        page_number = page_num + 1

        print(f"\n[Page {page_number}/{total_pages}]")

        # ── Render ──────────────────────────────────────────
        pil_img      = page_to_pil(page, dpi=200)
        img_w, img_h = pil_img.size

        # ── STEP 1: OCR ─────────────────────────────────────
        ocr_text, text_boxes = extract_ocr_text(pil_img)
        print(f"  OCR       : {len(ocr_text.splitlines())} lines")

        # ── STEP 2: Visual Region Detection ─────────────────
        yolo_boxes    = detect_yolo_regions(pil_img)
        contour_boxes = detect_contour_regions(pil_img, text_boxes)

        all_regions    = yolo_boxes + contour_boxes
        merged_regions = merge_boxes(all_regions, iou_threshold=0.05)

        real_regions = filter_real_image_regions(
            merged_regions, img_w, img_h
        )

        # ── Debug print ──────────────────────────────────────
        if real_regions:
            for i, (x1, y1, x2, y2) in enumerate(real_regions):
                w, h = x2 - x1, y2 - y1
                print(f"  Image {i+1:<3}: ({x1},{y1})→({x2},{y2})  size=({w}x{h})")
        else:
            print(f"  Images    : 0 detected")

        print(
            f"  YOLO:{len(yolo_boxes)} | "
            f"Contours:{len(contour_boxes)} | "
            f"Merged:{len(merged_regions)} | "
            f"Real images:{len(real_regions)}"
        )

        # Store per-page count
        page_image_counts[page_number] = len(real_regions)

        # ── STEP 3: Add image blocks (Gemma or placeholder) ──
        if use_gemma and real_regions:
            print(f"  Gemma     : sending {len(real_regions)} region(s) for analysis...")
            explanations = analyze_regions_with_gemma(pil_img, real_regions)

            for exp in explanations:
                image_block = ImageBlock(
                    description=exp["description"],
                    page=page_number
                )
                all_blocks.append(ContentBlock(
                    type=BlockType.IMAGE,
                    text=exp["description"],
                    page=page_number,
                    image=image_block,
                    metadata={
                        "source":     "gemma",
                        "has_images": True,
                        "region":     exp["region"],
                        "gemma_done": True
                    }
                ))
            # If no descriptions were obtained, fallback to placeholder
            if not explanations:
                print(f"  Gemma     : no descriptions received, adding placeholder")
                image_block = ImageBlock(
                    description=f"Page {page_number} has {len(real_regions)} detected image region(s) but Gemma did not return descriptions.",
                    page=page_number
                )
                all_blocks.append(ContentBlock(
                    type=BlockType.IMAGE,
                    text=f"[Page {page_number} contains {len(real_regions)} image/diagram region(s) – description unavailable]",
                    page=page_number,
                    image=image_block,
                    metadata={
                        "source":      "detection",
                        "has_images":  True,
                        "image_count": len(real_regions),
                        "regions":     [[x1, y1, x2, y2] for x1, y1, x2, y2 in real_regions],
                        "gemma_done":  False
                    }
                ))
        elif not use_gemma and real_regions:
            # Placeholder only (Gemma disabled)
            image_block = ImageBlock(
                description=f"Page {page_number} has {len(real_regions)} detected image region(s). Gemma analysis pending.",
                page=page_number
            )
            all_blocks.append(ContentBlock(
                type=BlockType.IMAGE,
                text=f"[Page {page_number} contains {len(real_regions)} image/diagram region(s)]",
                page=page_number,
                image=image_block,
                metadata={
                    "source":      "detection",
                    "has_images":  True,
                    "image_count": len(real_regions),
                    "regions":     [[x1, y1, x2, y2] for x1, y1, x2, y2 in real_regions],
                    "gemma_done":  False
                }
            ))
            print(
                f"  Gemma     : disabled — "
                f"{len(real_regions)} region(s) stored in metadata, "
                f"set use_gemma=True to analyse"
            )
        else:
            print(f"  Gemma     : skipped (no image regions)")

        # ── STEP 4: OCR Text Blocks ──────────────────────────
        text_blocks = parse_ocr_text_to_blocks(ocr_text, page_number)
        all_blocks.extend(text_blocks)

        # ── Page break ───────────────────────────────────────
        all_blocks.append(ContentBlock(
            type=BlockType.PAGE_BREAK,
            text=f"--- Page {page_number} ---",
            page=page_number,
            metadata={"document_name": file_name}
        ))

    doc.close()

    # ── Image summary ────────────────────────────────────────
    print("\n" + "─" * 45)
    print("IMAGE DETECTION SUMMARY")
    print("─" * 45)
    total_images = 0
    pages_with_images = []
    for pg, count in page_image_counts.items():
        status = f"{count} image(s)" if count > 0 else "no images"
        print(f"  Page {pg:2d} : {status}")
        total_images += count
        if count > 0:
            pages_with_images.append(pg)
    print("─" * 45)
    print(f"  Total images : {total_images}")
    print(f"  Pages with images : {pages_with_images if pages_with_images else 'None'}")
    print(f"  Gemma enabled     : {use_gemma}")
    if use_gemma and pages_with_images:
        print(f"  ✓ Gemma analysis performed for pages {pages_with_images}")
    elif not use_gemma and pages_with_images:
        print(f"  → Set use_gemma=True to describe images on pages {pages_with_images}")
    print("─" * 45)

    sections = build_sections(all_blocks, total_pages)

    return NormalizedDocument(
        file_name=file_name,
        total_pages=total_pages,
        pdf_type="scanned",
        sections=sections,
        blocks=all_blocks
    )