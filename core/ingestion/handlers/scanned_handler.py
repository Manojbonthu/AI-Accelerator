"""
core/ingestion/handlers/scanned_handler.py

Extracts content from scanned PDFs using:
- pymupdf: Convert pages to images
- pytesseract: OCR text extraction
- unstructured: Accurate layout detection (Text, Table, Figure)
- Google Gemma: Describe diagrams (via shared Gemma client)
"""

import os
import io
from typing import List, Optional
from PIL import Image
import pytesseract
import fitz  # pymupdf
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from core.schemas.models import (
    ContentBlock, TableBlock, ImageBlock,
    BlockType, NormalizedDocument
)
from core.ingestion.gemma_client import describe_image_with_gemma

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

TESSERACT_PATH = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ──────────────────────────────────────────────
# Main Extraction Function
# ──────────────────────────────────────────────

def extract_scanned(file_path: str, use_gemma: bool = True) -> NormalizedDocument:
    """
    Extract content from a scanned PDF.
    """
    file_name = os.path.basename(file_path)
    doc = fitz.open(file_path)
    total_pages = len(doc)
    print(f"  Converting PDF pages to images... {total_pages} pages")

    all_blocks = []

    for page_num in range(total_pages):
        page = doc[page_num]
        pix = page.get_pixmap(dpi=300)
        page_image = Image.open(io.BytesIO(pix.tobytes("png")))
        page_number = page_num + 1
        print(f"  Processing page {page_number}/{total_pages}...")

        # Add page break marker
        all_blocks.append(ContentBlock(
            type=BlockType.PAGE_BREAK,
            text=f"--- Page {page_number} ---",
            page=page_number
        ))

        # Step 1: Layout analysis using unstructured (Text, Table, Figure)
        layout_blocks = analyze_layout(page_image, page_number)

        # Step 2: OCR text for regions marked as Text
        ocr_text_blocks = extract_text_with_ocr(page_image, page_number)

        # Step 3: Merge layout info with OCR text
        merged_blocks = merge_layout_and_ocr(layout_blocks, ocr_text_blocks, page_number)

        # Step 4: For Figure/Image blocks, optionally call Gemma
        for block in merged_blocks:
            if block.type == BlockType.IMAGE and block.image and use_gemma:
                if block.image.image_bytes:
                    print(f"    Sending diagram to Gemma...")
                    description = describe_image_with_gemma(block.image.image_bytes)
                    if description:
                        block.image.description = description
                        block.text = description
                        print(f"    Gemma description: {description[:100]}...")

        all_blocks.extend(merged_blocks)

    doc.close()
    return NormalizedDocument(
        file_name=file_name,
        total_pages=total_pages,
        pdf_type="scanned",
        blocks=all_blocks
    )


# ──────────────────────────────────────────────
# Layout Analysis with unstructured
# ──────────────────────────────────────────────

def _safe_get_coordinates(el) -> Optional[tuple]:
    """
    Safely extract (x1, y1, x2, y2) from an unstructured element's metadata.
    Handles both tuple/list and CoordinatesMetadata objects.
    """
    if not hasattr(el.metadata, 'coordinates'):
        return None
    c = el.metadata.coordinates
    if c is None:
        return None
    # If it's already a sequence of 4 numbers
    if isinstance(c, (list, tuple)) and len(c) == 4:
        return tuple(c)
    # If it's a CoordinatesMetadata object with .points
    if hasattr(c, 'points') and hasattr(c.points, 'x1'):
        pts = c.points
        return (pts.x1, pts.y1, pts.x2, pts.y2)
    # fallback: try to convert to tuple if it's iterable
    try:
        return tuple(c)
    except Exception:
        return None


def analyze_layout(image: Image.Image, page_num: int) -> List[ContentBlock]:
    """
    Use unstructured to detect layout regions (Text, Table, Figure).
    Falls back to gap detection if unstructured is not available.
    """
    try:
        from unstructured.partition.image import partition_image
        from unstructured.documents.elements import ElementType

        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        elements = partition_image(
            file=img_bytes,
            include_page_breaks=False,
            strategy="hi_res",
        )

        blocks = []
        for el in elements:
            el_type = el.category
            coords = _safe_get_coordinates(el)

            if el_type == ElementType.TABLE:
                blocks.append(ContentBlock(
                    type=BlockType.TABLE,
                    text=el.text,
                    page=page_num,
                    table=TableBlock(
                        headers=[],
                        rows=[],
                        page=page_num
                    ),
                    metadata={"source": "unstructured", "bbox": str(coords)}
                ))
            elif el_type in (ElementType.IMAGE, ElementType.FIGURE):
                if coords:
                    x1, y1, x2, y2 = coords
                    cropped = image.crop((x1, y1, x2, y2))
                    buf = io.BytesIO()
                    cropped.save(buf, format='PNG')
                    blocks.append(ContentBlock(
                        type=BlockType.IMAGE,
                        text="",
                        page=page_num,
                        image=ImageBlock(
                            image_bytes=buf.getvalue(),
                            mime_type="image/png",
                            page=page_num
                        ),
                        metadata={"source": "unstructured", "bbox": str(coords)}
                    ))
            else:
                blocks.append(ContentBlock(
                    type=BlockType.PARAGRAPH,
                    text=el.text,
                    page=page_num,
                    metadata={"source": "unstructured", "bbox": str(coords)}
                ))
        return blocks

    except ImportError:
        print("  WARNING: 'unstructured' not installed. Falling back to gap detection.")
        return detect_and_describe_diagrams_fallback(image, page_num, use_gemma=False)


def merge_layout_and_ocr(
    layout_blocks: List[ContentBlock],
    ocr_text_blocks: List[ContentBlock],
    page_num: int
) -> List[ContentBlock]:
    """Replace layout TEXT blocks with high‑quality OCR text."""
    ocr_idx = 0
    merged = []
    for block in layout_blocks:
        if block.type == BlockType.PARAGRAPH:
            if ocr_idx < len(ocr_text_blocks):
                merged.append(ocr_text_blocks[ocr_idx])
                ocr_idx += 1
            else:
                merged.append(block)
        else:
            merged.append(block)
    return merged


# ──────────────────────────────────────────────
# OCR Text Extraction
# ──────────────────────────────────────────────

def extract_text_with_ocr(image: Image.Image, page_num: int) -> List[ContentBlock]:
    """Run OCR on a page image and return text blocks grouped into paragraphs."""
    ocr_data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config='--psm 6'
    )

    blocks = []
    current_paragraph = []
    current_confidences = []
    last_y = -1
    last_bottom = -1
    paragraph_gap_threshold = 30

    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        conf = int(ocr_data['conf'][i]) if ocr_data['conf'][i] != '-1' else 0

        if not text or conf < 10:
            if current_paragraph and last_bottom > 0:
                current_y = ocr_data['top'][i]
                gap = current_y - last_bottom
                if gap > paragraph_gap_threshold:
                    combined_text = " ".join(current_paragraph)
                    avg_confidence = (sum(current_confidences) / len(current_confidences)
                                      if current_confidences else 0)
                    blocks.append(ContentBlock(
                        type=BlockType.PARAGRAPH,
                        text=combined_text,
                        page=page_num,
                        metadata={
                            "ocr_confidence": round(avg_confidence, 2),
                            "source": "ocr"
                        }
                    ))
                    current_paragraph = []
                    current_confidences = []
                    last_y = -1
                    last_bottom = -1
            continue

        y = ocr_data['top'][i]
        h = ocr_data['height'][i]
        bottom = y + h

        if current_paragraph and last_y != -1:
            y_gap = abs(y - last_y)
            if y_gap > paragraph_gap_threshold:
                combined_text = " ".join(current_paragraph)
                avg_confidence = (sum(current_confidences) / len(current_confidences)
                                  if current_confidences else 0)
                blocks.append(ContentBlock(
                    type=BlockType.PARAGRAPH,
                    text=combined_text,
                    page=page_num,
                    metadata={
                        "ocr_confidence": round(avg_confidence, 2),
                        "source": "ocr"
                    }
                ))
                current_paragraph = []
                current_confidences = []

        current_paragraph.append(text)
        current_confidences.append(conf)
        last_y = y
        last_bottom = bottom

    if current_paragraph:
        combined_text = " ".join(current_paragraph)
        avg_confidence = (sum(current_confidences) / len(current_confidences)
                          if current_confidences else 0)
        blocks.append(ContentBlock(
            type=BlockType.PARAGRAPH,
            text=combined_text,
            page=page_num,
            metadata={
                "ocr_confidence": round(avg_confidence, 2),
                "source": "ocr"
            }
        ))

    return blocks


# ──────────────────────────────────────────────
# Fallback Gap Detection
# ──────────────────────────────────────────────

def detect_and_describe_diagrams_fallback(
    image: Image.Image,
    page_num: int,
    use_gemma: bool = True
) -> List[ContentBlock]:
    """Original gap‑based detection, used as fallback."""
    blocks = []
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config='--psm 6')
    image_width = image.width

    text_regions = []
    for i in range(len(ocr_data['text'])):
        if ocr_data['text'][i].strip() and int(ocr_data['conf'][i]) > 30:
            text_regions.append({
                'x': ocr_data['left'][i],
                'y': ocr_data['top'][i],
                'width': ocr_data['width'][i],
                'height': ocr_data['height'][i],
                'text': ocr_data['text'][i].strip()
            })

    if len(text_regions) >= 2:
        text_regions.sort(key=lambda r: r['y'])
        for i in range(len(text_regions) - 1):
            current_bottom = text_regions[i]['y'] + text_regions[i]['height']
            next_top = text_regions[i + 1]['y']
            gap = next_top - current_bottom
            if gap > 100:
                diagram_region = image.crop((0, current_bottom + 5, image_width, next_top - 5))
                if has_content(diagram_region):
                    img_bytes = io.BytesIO()
                    diagram_region.save(img_bytes, format='PNG')
                    img_bytes = img_bytes.getvalue()
                    caption = find_nearby_caption(text_regions, i)
                    description = None
                    if use_gemma:
                        description = describe_image_with_gemma(img_bytes)
                    blocks.append(ContentBlock(
                        type=BlockType.IMAGE,
                        text=description or caption or "[Diagram found on page]",
                        page=page_num,
                        image=ImageBlock(
                            image_bytes=img_bytes,
                            mime_type="image/png",
                            caption=caption,
                            description=description,
                            page=page_num
                        ),
                        metadata={"source": "gap_detection", "gap_size": gap}
                    ))
    return blocks


def has_content(image: Image.Image) -> bool:
    import numpy as np
    img_array = np.array(image.convert('L'))
    return np.std(img_array) > 20


def find_nearby_caption(text_regions: List[dict], diagram_index: int) -> Optional[str]:
    if diagram_index + 1 < len(text_regions):
        next_text = text_regions[diagram_index + 1]['text']
        if next_text.lower().startswith(('fig', 'figure', 'table', 'image', 'diagram')):
            return next_text
    return None


# ──────────────────────────────────────────────
# Helper: Test Gemma Connection
# ──────────────────────────────────────────────

def test_gemma_connection() -> bool:
    """Test if Gemma API is accessible with current key."""
    try:
        from google import genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ GOOGLE_API_KEY not set in .env")
            return False
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.getenv("GEMMA_MODEL", "gemma-2-flash"),
            contents="Say 'API connection successful' if you can read this."
        )
        print(f"✅ Gemma API connected! Response: {response.text[:50]}")
        return True
    except Exception as e:
        print(f"❌ Gemma API connection failed: {e}")
        return False