"""
core/ingestion/pdf_detector.py

Detects PDF type per page and overall:
- Digital: Page has real text layer
- Scanned: Page is just an image (no text)
- Mixed: Some pages digital, some scanned
"""

import fitz  # pymupdf
from typing import Tuple, List


def detect_pdf_type(file_path: str, text_threshold: int = 50) -> Tuple[str, List[str]]:
    """
    Analyze a PDF and determine its type.
    
    Args:
        file_path: Path to the PDF file
        text_threshold: Minimum characters to consider a page as digital
    
    Returns:
        Tuple of (overall_type, page_types)
        - overall_type: "digital", "scanned", or "mixed"
        - page_types: List of types per page ["digital", "scanned", ...]
    """
    doc = fitz.open(file_path)
    page_types = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        
        # Count actual characters (ignore whitespace-only)
        char_count = len(text)
        
        if char_count >= text_threshold:
            page_types.append("digital")
        else:
            page_types.append("scanned")
    
    doc.close()
    
    # Determine overall type
    if all(t == "digital" for t in page_types):
        overall_type = "digital"
    elif all(t == "scanned" for t in page_types):
        overall_type = "scanned"
    else:
        overall_type = "mixed"
    
    return overall_type, page_types


def get_page_count(file_path: str) -> int:
    """Get total number of pages in a PDF."""
    doc = fitz.open(file_path)
    count = len(doc)
    doc.close()
    return count