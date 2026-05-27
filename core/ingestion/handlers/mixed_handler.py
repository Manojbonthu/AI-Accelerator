"""
core/ingestion/handlers/mixed_handler.py

Handles mixed PDFs (some pages digital, some scanned).
Routes each page to the appropriate extraction method.
Now uses shared Gemma client for both digital and scanned images.
"""

import os
import io
from typing import List
from PIL import Image
import fitz
from dotenv import load_dotenv

load_dotenv()

from core.schemas.models import (
    ContentBlock, TableBlock, ImageBlock,
    BlockType, NormalizedDocument
)
from core.ingestion.pdf_detector import detect_pdf_type
from core.ingestion.handlers.digital_handler import (
    process_text_block, process_image_block, extract_tables_from_page
)
from core.ingestion.handlers.scanned_handler import (
    analyze_layout,
    merge_layout_and_ocr,
    extract_text_with_ocr,
)
from core.ingestion.gemma_client import describe_image_with_gemma


def extract_mixed(file_path: str, use_gemma: bool = True) -> NormalizedDocument:
    """
    Extract content from a mixed PDF (digital + scanned pages).
    
    Args:
        file_path: Path to the mixed PDF file
        use_gemma: Whether to use Gemma API for diagram descriptions
    
    Returns:
        NormalizedDocument with all extracted content blocks
    """
    file_name = os.path.basename(file_path)
    
    # Detect per-page types
    overall_type, page_types = detect_pdf_type(file_path)
    
    # Open PDF for both digital and scanned extraction
    doc = fitz.open(file_path)
    total_pages = len(page_types)
    
    all_blocks = []
    section_stack = []   # Track heading hierarchy
    section_counter = 0

    for page_num in range(total_pages):
        page_type = page_types[page_num]
        page_number = page_num + 1
        
        # Add page break marker
        all_blocks.append(ContentBlock(
            type=BlockType.PAGE_BREAK,
            text=f"--- Page {page_number} ---",
            page=page_number
        ))

        if page_type == "digital":
            # Extract digital page using digital handler logic, with Gemma option
            digital_blocks = extract_digital_page(
                doc, page_num, page_number, section_counter, use_gemma
            )
            # Update section tracking based on headings found
            for block in digital_blocks:
                if block.type == BlockType.HEADING and block.level:
                    section_counter += 1
                    block.section_id = f"section_{section_counter}"
                    while section_stack and section_stack[-1][0] >= block.level:
                        section_stack.pop()
                    section_stack.append((block.level, block.text, block.section_id))
                if section_stack:
                    block.section_id = section_stack[-1][2]
                all_blocks.append(block)

        else:  # scanned page
            # Render scanned page to PIL image using pymupdf
            page = doc[page_num]
            pix = page.get_pixmap(dpi=300)
            page_image = Image.open(io.BytesIO(pix.tobytes("png")))
            
            # Layout analysis with unstructured
            layout_blocks = analyze_layout(page_image, page_number)
            # OCR text extraction
            ocr_text_blocks = extract_text_with_ocr(page_image, page_number)
            # Merge layout with OCR
            scanned_blocks = merge_layout_and_ocr(layout_blocks, ocr_text_blocks, page_number)
            
            # Process figures with Gemma and assign section context
            for block in scanned_blocks:
                if section_stack:
                    block.section_id = section_stack[-1][2]
                
                # If it's an image/figure and Gemma is enabled, get description
                if block.type == BlockType.IMAGE and block.image and use_gemma:
                    if block.image.image_bytes:
                        desc = describe_image_with_gemma(block.image.image_bytes)
                        if desc:
                            block.image.description = desc
                            block.text = desc  # use description as chunk text
                
                all_blocks.append(block)

    doc.close()
    
    return NormalizedDocument(
        file_name=file_name,
        total_pages=total_pages,
        pdf_type="mixed",
        blocks=all_blocks
    )


def extract_digital_page(
    doc: fitz.Document,
    page_num: int,
    page_number: int,
    section_counter: int,
    use_gemma: bool = True
) -> List[ContentBlock]:
    """
    Extract content from a single digital page.
    Now passes use_gemma to process_image_block for digital images.
    """
    page = doc[page_num]
    blocks_data = page.get_text("dict")["blocks"]
    
    page_blocks = []
    
    for block in blocks_data:
        if block["type"] == 0:  # Text
            text_blocks = process_text_block(block, page_number, section_counter)
            page_blocks.extend(text_blocks)
        elif block["type"] == 1:  # Image
            image_blocks = process_image_block(block, doc, page_number, use_gemma)
            page_blocks.extend(image_blocks)
    
    # Also extract tables using pymupdf's table detection
    table_blocks = extract_tables_from_page(doc, page_num)
    page_blocks.extend(table_blocks)
    
    return page_blocks