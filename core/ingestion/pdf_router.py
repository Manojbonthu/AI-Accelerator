"""
core/ingestion/pdf_router.py

Main entry point - one function for the entire pipeline.
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

from core.ingestion.pdf_detector import detect_pdf_type
from core.ingestion.handlers.digital_handler import extract_digital
from core.ingestion.handlers.scanned_handler import extract_scanned
from core.ingestion.chunker import chunk_document
from core.ingestion.storage import QdrantStorage
from core.schemas.models import Chunk


def extract_pdf(
    file_path: str,
    store_in_qdrant: bool = False,
    use_gemma: bool = True
) -> Dict[str, Any]:
    """Main entry point for PDF extraction pipeline."""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")
    
    print(f"\n{'='*50}")
    print(f"Processing: {os.path.basename(file_path)}")
    
    # Step 1: Detect
    pdf_type, page_types = detect_pdf_type(file_path)
    print(f"Detected: {pdf_type}")
    
    # Step 2: Extract (mixed PDFs are routed to scanned handler for simplicity)
    if pdf_type == "digital":
        normalized_doc = extract_digital(file_path, use_gemma=use_gemma)
    else:  # scanned or mixed – scanned_handler works for both
        normalized_doc = extract_scanned(file_path, use_gemma=use_gemma)
    
    # Step 3: Chunk
    chunks = chunk_document(normalized_doc)
    print(f"Created {len(chunks)} chunks")
    
    # Step 4: Optionally store
    storage_info = None
    if store_in_qdrant:
        try:
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_PORT", "6333"))
            storage = QdrantStorage(host=host, port=port)
            storage.create_collection()
            count = storage.store_chunks(chunks)
            storage_info = {"stored": True, "points": count}
            print(f"Stored {count} chunks in Qdrant")
        except Exception as e:
            storage_info = {"stored": False, "error": str(e)}
            print(f"Qdrant error: {e}")
    
    return {
        "file_name": normalized_doc.file_name,
        "pdf_type": pdf_type,
        "total_pages": normalized_doc.total_pages,
        "total_chunks": len(chunks),
        "page_types": page_types,
        "storage": storage_info,
        "chunks": [chunk.to_dict() for chunk in chunks]
    }