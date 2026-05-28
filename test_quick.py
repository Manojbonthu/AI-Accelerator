"""
check_pages.py (or test_quick.py)
Run: python test_quick.py
"""

import fitz  # pymupdf

def analyze_page(page):
    """Return a dict of metrics for a single page."""
    text = page.get_text().strip()
    text_length = len(text)
    
    # Raster images
    raster_count = len(page.get_images(full=True))
    
    # Vector drawings
    vector_count = len(page.get_drawings())
    
    # Annotations (convert generator to list to get length)
    annot_count = len(list(page.annots())) if hasattr(page, 'annots') else 0
    
    # Links
    link_count = len(page.get_links())
    
    # Image coverage ratio (only for raster images)
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height
    image_area = 0.0
    if raster_count > 0:
        # get_images(full=True) returns list of tuples; each tuple contains image reference
        for img_ref in page.get_images(full=True):
            # img_ref is a tuple; the first element is the xref
            rects = page.get_image_rects(img_ref)  # pass the whole tuple or just xref? It expects the image object as returned by get_images()
            for r in rects:
                image_area += r.width * r.height
    image_coverage = image_area / page_area if page_area > 0 else 0.0
    
    return {
        "text_length": text_length,
        "raster_images": raster_count,
        "vector_drawings": vector_count,
        "annotations": annot_count,
        "links": link_count,
        "image_coverage_ratio": round(image_coverage, 4)
    }


def detect_page_types(file_path: str, text_threshold: int = 5):
    """
    Returns: (overall_type, list_of_page_types, list_of_metrics)
    """
    doc = fitz.open(file_path)
    page_types = []
    metrics_list = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        char_count = len(text)
        if char_count >= text_threshold:
            page_types.append("digital")
        else:
            page_types.append("scanned")
        
        # Get detailed metrics
        metrics = analyze_page(page)
        metrics_list.append(metrics)
    
    doc.close()
    
    if all(t == "digital" for t in page_types):
        overall = "digital"
    elif all(t == "scanned" for t in page_types):
        overall = "scanned"
    else:
        overall = "mixed"
    
    return overall, page_types, metrics_list


if __name__ == "__main__":
    pdf_path = "tests/sample_docs/ano_digital_40pages.pdf"   # change to your file
    threshold = 5   # change to 1 if needed
    
    overall, types, metrics = detect_page_types(pdf_path, text_threshold=threshold)
    print(f"Overall type: {overall} (using threshold {threshold})\n")
    print("Page | Type     | TextLen | Rast | Vect | Annot | Links | Cov%")
    print("-----|----------|---------|------|------|-------|-------|------")
    for i, (t, m) in enumerate(zip(types, metrics), start=1):
        print(f"{i:3d} | {t:8s} | {m['text_length']:7d} | {m['raster_images']:4d} | {m['vector_drawings']:4d} | {m['annotations']:5d} | {m['links']:5d} | {m['image_coverage_ratio']:.3f}")
    
    # Totals
    total_raster = sum(m['raster_images'] for m in metrics)
    total_vector = sum(m['vector_drawings'] for m in metrics)
    total_annot = sum(m['annotations'] for m in metrics)
    total_links = sum(m['links'] for m in metrics)
    print("\nTotals:")
    print(f"  Raster images: {total_raster}")
    print(f"  Vector drawings: {total_vector}")
    print(f"  Annotations: {total_annot}")
    print(f"  Links: {total_links}")