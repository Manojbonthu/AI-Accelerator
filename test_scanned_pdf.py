"""
test_scanned_pdf.py
Test the scanned PDF handler on PublicWaterMassMailing.pdf.
"""

import sys
import json
import time

sys.path.insert(0, '.')

from core.ingestion.pdf_analyzer import analyze_pdf


# ─── Configuration ─────────────────────────
pdf_path    = "tests/sample_docs/PublicWaterMassMailing.pdf"
use_gemma   = True        # set True only after OCR works correctly
output_json = "new_water_new.json"
domain      = "general"
# ────────────────────────────────────────────


def make_serializable(obj):
    """
    Recursively convert any non-serializable objects
    (Pydantic models, dataclasses, custom objects) to dicts.
    Handles nested lists and dicts.
    """
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_serializable(i) for i in obj]

    if isinstance(obj, bytes):
        return "<bytes>"

    if hasattr(obj, '__dict__'):
        return make_serializable(vars(obj))

    return obj


def print_chunk_preview(chunks: list, n: int = 3):
    print("\n" + "=" * 60)
    print(f"FIRST {n} CHUNKS (of {len(chunks)})")
    print("=" * 60)

    for i, chunk in enumerate(chunks[:n]):
        print(f"\n--- Chunk {i} ---")
        print(
            f"  Type    : {chunk.get('chunk_type', 'N/A')}\n"
            f"  Section : {chunk.get('section', 'N/A')}\n"
            f"  Pages   : {chunk.get('page_start', '?')}"
            f"-{chunk.get('page_end', '?')}"
        )

        content = chunk.get('content', '') or ''
        preview = content[:120].replace('\n', ' ')
        print(f"  Content : {preview}...")

        tables = chunk.get('tables') or []
        if tables:
            print(f"  Tables  : {len(tables)}")

        figures = chunk.get('figures') or []
        if figures:
            print(f"  Figures : {len(figures)}")
            for j, fig in enumerate(figures):
                if isinstance(fig, dict):
                    desc = (fig.get('description') or '')[:80]
                else:
                    desc = str(fig)[:80]
                print(f"    Figure {j + 1}: {desc}...")

        rel = chunk.get('relationships') or {}
        print(f"  Prev    : {rel.get('previous_chunk_id', 'None')}")
        print(f"  Next    : {rel.get('next_chunk_id', 'None')}")

    if len(chunks) > n:
        print(f"\n... and {len(chunks) - n} more chunks (see {output_json})")


def run_test():
    print(f"Analyzing: {pdf_path}")
    print(f"Gemma    : {use_gemma}")
    print(f"Domain   : {domain}\n")

    t_start = time.time()

    try:
        result = analyze_pdf(
            pdf_path,
            use_gemma=use_gemma,
            domain=domain
        )
    except Exception as e:
        print(f"ERROR during analyze_pdf: {e}")
        raise

    t_total = time.time() - t_start

    # ── Validate result keys ──────────────────────────────
    required_keys = [
        'chunks', 'file_name', 'pdf_type',
        'total_pages', 'total_chunks', 'summary_text'
    ]
    missing = [k for k in required_keys if k not in result]
    if missing:
        print(f"WARNING: result missing keys: {missing}")

    chunks = result.get('chunks') or []

    # ── Stats ─────────────────────────────────────────────
    total_tables  = sum(1 for c in chunks if c.get('tables'))
    total_figures = sum(1 for c in chunks if c.get('figures'))

    print(f"Total time    : {t_total:.2f}s")
    print("─" * 40)
    print(f"File          : {result.get('file_name', 'N/A')}")
    print(f"Type          : {result.get('pdf_type', 'N/A')}")
    print(f"Pages         : {result.get('total_pages', 'N/A')}")
    print(f"Total chunks  : {result.get('total_chunks', len(chunks))}")
    print(f"With tables   : {total_tables}")
    print(f"With figures  : {total_figures}")

    summary = result.get('summary_text', '')
    if summary:
        print(f"\n{summary}")

    # ── Serialize & Save ──────────────────────────────────
    try:
        serializable_chunks = make_serializable(chunks)

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(serializable_chunks, f, indent=2, ensure_ascii=False)

        print(f"\nSaved {len(chunks)} chunks → {output_json}")

    except Exception as e:
        print(f"ERROR saving JSON: {e}")
        raise

    # ── Preview ───────────────────────────────────────────
    print_chunk_preview(chunks, n=3)


if __name__ == "__main__":
    run_test()