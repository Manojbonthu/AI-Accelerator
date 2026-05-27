"""Quick test for digital PDF with Gemma descriptions."""
import sys, os
sys.path.insert(0, '.')

from core.ingestion.pdf_analyzer import analyze_pdf

filepath = "tests/sample_docs/c4611_sample_explain.pdf"

print("Analyzing with Gemma descriptions (this may take a moment)...")
result = analyze_pdf(filepath, use_gemma=True)

print(result["summary_text"])
print()

print("Detailed Page Classification:")
header = f"{'Page':<6} {'Digital Text':<14} {'Images/Diagrams':<17} {'Tables':<8} {'Blank':<7} Description"
print(header)
print("-" * len(header))
for p in result["detailed_summary"]:
    print(f"{p['page']:<6} {p['digital_text']:<14} {p['image']:<17} {p['table']:<8} {p['blank']:<7} {p['description'][:100]}")
print()
print(f"Total chunks: {result['total_chunks']}")