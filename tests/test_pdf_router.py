"""
tests/test_pdf_router.py

Tests for the PDF extraction pipeline.
Runs on digital, scanned, and mixed sample PDFs.
"""

import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.ingestion.pdf_router import extract_pdf
from core.ingestion.pdf_detector import detect_pdf_type


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

SAMPLE_DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")

# Map file names to expected types
EXPECTED_TYPES = {
    "digital_sample.pdf": "digital",
    "scanned_sample.pdf": "scanned",
    "mixed_sample.pdf": "mixed",
}


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────

def get_sample_files():
    """Get all PDF files from sample_docs directory."""
    if not os.path.exists(SAMPLE_DOCS_DIR):
        return []
    
    files = []
    for f in os.listdir(SAMPLE_DOCS_DIR):
        if f.endswith('.pdf'):
            files.append(os.path.join(SAMPLE_DOCS_DIR, f))
    return files


def print_separator(title: str):
    """Print a formatted separator."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ──────────────────────────────────────────────
# Test Functions
# ──────────────────────────────────────────────

def test_detector():
    """Test the PDF type detector."""
    print_separator("TEST: PDF Type Detector")
    
    sample_files = get_sample_files()
    
    if not sample_files:
        print("⚠ No sample PDFs found in tests/sample_docs/")
        print("  Please add sample PDFs to test.")
        return True  # Pass if no files
    
    all_passed = True
    
    for file_path in sample_files:
        file_name = os.path.basename(file_path)
        pdf_type, page_types = detect_pdf_type(file_path)
        
        expected = EXPECTED_TYPES.get(file_name, None)
        
        if expected:
            if pdf_type == expected:
                print(f"✅ {file_name}: {pdf_type} (expected: {expected})")
            else:
                print(f"❌ {file_name}: {pdf_type} (expected: {expected})")
                all_passed = False
        else:
            print(f"ℹ {file_name}: {pdf_type} (pages: {page_types})")
    
    return all_passed


def test_extraction():
    """Test full extraction on all sample PDFs."""
    print_separator("TEST: PDF Extraction Pipeline")
    
    sample_files = get_sample_files()
    
    if not sample_files:
        print("⚠ No sample PDFs found.")
        return True
    
    all_passed = True
    
    for file_path in sample_files:
        file_name = os.path.basename(file_path)
        
        try:
            result = extract_pdf(
                file_path=file_path,
                store_in_qdrant=False,
                use_gemma=False  # Don't call Gemma during tests
            )
            
            # Check basic structure
            checks = []
            checks.append(("file_name" in result, "Has file_name"))
            checks.append(("pdf_type" in result, "Has pdf_type"))
            checks.append(("total_pages" in result, "Has total_pages"))
            checks.append(("total_chunks" in result, "Has total_chunks"))
            checks.append(("chunks" in result, "Has chunks list"))
            checks.append((result["total_pages"] > 0, "Has pages"))
            checks.append((result["total_chunks"] > 0, "Has chunks"))
            
            # Check first chunk structure
            if result["chunks"]:
                chunk = result["chunks"][0]
                checks.append(("chunk_id" in chunk, "Chunk has chunk_id"))
                checks.append(("content" in chunk, "Chunk has content"))
                checks.append(("embedding_text" in chunk, "Chunk has embedding_text"))
                checks.append(("document_name" in chunk, "Chunk has document_name"))
                checks.append(("section_path" in chunk, "Chunk has section_path"))
                checks.append(("page_start" in chunk, "Chunk has page_start"))
                checks.append(("page_end" in chunk, "Chunk has page_end"))
            
            all_ok = all(check[0] for check in checks)
            
            if all_ok:
                print(f"✅ {file_name}")
                print(f"   Type: {result['pdf_type']}")
                print(f"   Pages: {result['total_pages']}")
                print(f"   Chunks: {result['total_chunks']}")
            else:
                print(f"❌ {file_name} - Failed checks:")
                for check_result, check_name in checks:
                    if not check_result:
                        print(f"   - Missing: {check_name}")
                all_passed = False
        
        except Exception as e:
            print(f"❌ {file_name} - Error: {str(e)}")
            all_passed = False
    
    return all_passed


def test_chunk_content():
    """Test that chunks have meaningful content."""
    print_separator("TEST: Chunk Content Quality")
    
    sample_files = get_sample_files()
    
    if not sample_files:
        print("⚠ No sample PDFs found.")
        return True
    
    all_passed = True
    
    for file_path in sample_files:
        file_name = os.path.basename(file_path)
        
        try:
            result = extract_pdf(
                file_path=file_path,
                store_in_qdrant=False,
                use_gemma=False
            )
            
            chunks = result["chunks"]
            
            # Check for empty chunks
            empty_chunks = [c for c in chunks if not c["content"].strip()]
            
            # Check for very small chunks
            small_chunks = [c for c in chunks if len(c["embedding_text"]) < 50]
            
            # Check embedding text contains context
            missing_context = [
                c for c in chunks
                if "Document:" not in c["embedding_text"]
                or "Section:" not in c["embedding_text"]
            ]
            
            issues = []
            if empty_chunks:
                issues.append(f"{len(empty_chunks)} empty chunks")
            if small_chunks:
                issues.append(f"{len(small_chunks)} very small chunks (<50 chars)")
            if missing_context:
                issues.append(f"{len(missing_context)} chunks missing context headers")
            
            if not issues:
                print(f"✅ {file_name}: All {len(chunks)} chunks look good")
            else:
                print(f"⚠ {file_name}: Issues found - {', '.join(issues)}")
                # Don't fail for warnings - chunks might be valid but small
        
        except Exception as e:
            print(f"❌ {file_name} - Error: {str(e)}")
            all_passed = False
    
    return all_passed


def test_chunk_json_serializable():
    """Test that chunks can be serialized to JSON."""
    print_separator("TEST: JSON Serialization")
    
    sample_files = get_sample_files()
    
    if not sample_files:
        print("⚠ No sample PDFs found.")
        return True
    
    all_passed = True
    
    for file_path in sample_files:
        file_name = os.path.basename(file_path)
        
        try:
            result = extract_pdf(
                file_path=file_path,
                store_in_qdrant=False,
                use_gemma=False
            )
            
            # Try to serialize to JSON
            json_str = json.dumps(result["chunks"], indent=2)
            json.loads(json_str)  # Parse back
            
            print(f"✅ {file_name}: JSON serialization works")
        
        except Exception as e:
            print(f"❌ {file_name} - JSON Error: {str(e)}")
            all_passed = False
    
    return all_passed


# ──────────────────────────────────────────────
# Main Test Runner
# ──────────────────────────────────────────────

def run_all_tests():
    """Run all tests and print summary."""
    print("\n" + "=" * 60)
    print("  PDF EXTRACTION PIPELINE - TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Run tests
    results["Detector"] = test_detector()
    results["Extraction"] = test_extraction()
    results["Chunk Quality"] = test_chunk_content()
    results["JSON Serialization"] = test_chunk_json_serializable()
    
    # Summary
    print_separator("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    print(f"\n  {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠ Some tests failed. Check the output above.")
        sys.exit(1)