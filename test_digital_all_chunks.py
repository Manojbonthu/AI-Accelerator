"""
Digital PDF – print all chunks with stats, timings, and relationships.
"""
import sys
import json
import time

sys.path.insert(0, '.')

from core.ingestion.pdf_analyzer import analyze_pdf

# ─── Configuration ─────────────────────────
pdf_path = "tests/sample_docs/ano_digital_40pages.pdf"
use_gemma = True   # Set to True only if Gemini API is configured
output_json = "all_chunks_40pagesssss.json"
# ────────────────────────────────────────────

print(f"Analyzing {pdf_path} (use_gemma={use_gemma})...")
t_start = time.time()
result = analyze_pdf(pdf_path, use_gemma=use_gemma, domain="general")
t_total = time.time() - t_start

# Document Stats
total_tables = sum(1 for c in result['chunks'] if c.get('tables'))
total_figures = sum(1 for c in result['chunks'] if c.get('figures'))

print(f"\nTotal analysis time: {t_total:.2f}s")
print("--- Document Stats ---")
print(f"File: {result['file_name']}")
print(f"Type: {result['pdf_type']}")
print(f"Pages: {result['total_pages']}")
print(f"Total chunks: {result['total_chunks']}")
print(f"Chunks with tables: {total_tables}")
print(f"Chunks with figures: {total_figures}")

# Per‑page summary
print("\n" + result["summary_text"])

# Save all chunks to JSON
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(result["chunks"], f, indent=2, ensure_ascii=False)
print(f"\n✅ All {len(result['chunks'])} chunks saved to {output_json}")

# Print each chunk with details
print("\n" + "="*60)
print("ALL CHUNKS")
print("="*60)
for i, chunk in enumerate(result['chunks']):
    print(f"\n--- Chunk {i} ---")
    print(f"  Type: {chunk['chunk_type']}, Section: {chunk['section']}, Pages: {chunk['page_start']}-{chunk['page_end']}")
    # Safely print content (handle empty or None)
    content_preview = chunk.get('content', '')[:120].replace('\n', ' ') if chunk.get('content') else ''
    print(f"  Content (first 120 chars): {content_preview}...")
    
    if chunk.get('tables'):
        print(f"  📊 Tables: {len(chunk['tables'])}")
        for tbl in chunk['tables']:
            print(f"    Headers: {tbl.get('headers', [])}, Rows: {len(tbl.get('rows', []))}")
    
    if chunk.get('figures'):
        print(f"  🖼️ Figures: {len(chunk['figures'])}")
        for j, fig in enumerate(chunk['figures']):
            # Safely get description, default to empty string if None
            desc = (fig.get('description') or '')[:80]
            print(f"    Figure {j+1}: {desc}...")
    
    rel = chunk.get('relationships', {})
    print(f"  ⛓️ Previous chunk: {rel.get('previous_chunk_id', 'None')}")
    print(f"  ⛓️ Next chunk: {rel.get('next_chunk_id', 'None')}")