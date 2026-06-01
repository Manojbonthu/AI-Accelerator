"""
PDF detection module – checks if pages contain selectable text.
Returns overall type ("digital", "scanned", "mixed") and per‑page types.
"""

import fitz  # PyMuPDF
from typing import Tuple, List


def detect_pdf_type(file_path: str) -> Tuple[str, List[str]]:
    """
    Detect whether a PDF is digital (all pages have selectable text),
    scanned (no page has selectable text), or mixed (some pages have text, some do not).

    Args:
        file_path: Path to the PDF file.

    Returns:
        Tuple: (overall_type, list_of_page_types)
        overall_type: "digital", "scanned", or "mixed"
        page_types: list of "digital" or "scanned" for each page
    """
    doc = fitz.open(file_path)
    total_pages = len(doc)
    page_types = []

    print("\nPage-by-page classification:")
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text().strip()
        # Lowered threshold – any page with at least 5 characters is considered digital
        if len(text) > 5:
            page_types.append("digital")
            print(f"Page {page_num+1}: digital ({len(text)} chars)")
        else:
            page_types.append("scanned")
            print(f"Page {page_num+1}: scanned (only {len(text)} chars)")

    doc.close()

    if all(t == "digital" for t in page_types):
        overall = "digital"
    elif all(t == "scanned" for t in page_types):
        overall = "scanned"
    else:
        overall = "mixed"

    print(f"\nOverall PDF type: {overall}")
    return overall, page_types