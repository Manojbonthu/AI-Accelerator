"""
test_gemma_images.py

Extracts all images from a digital PDF and sends each to Gemma.
Prints the API response (or error) for every image.
"""

import sys, os, io
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

import fitz  # pymupdf
from core.ingestion.gemma_client import describe_image_with_gemma

# ─── Configuration ───────────────────────
PDF_PATH = "tests/sample_docs/c4611_sample_explain.pdf"   # change if needed
MAX_IMAGE_SIZE = (1024, 1024)             # resize larger images to avoid timeouts
# ─────────────────────────────────────────

def resize_image_bytes(image_bytes, max_size=MAX_IMAGE_SIZE):
    """Resize image if larger than max_size while keeping aspect ratio."""
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes))
    if img.width > max_size[0] or img.height > max_size[1]:
        img.thumbnail(max_size)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    return image_bytes

def main():
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not set in .env file.")
        return

    doc = fitz.open(PDF_PATH)
    total_images = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images()
        if not images:
            continue

        print(f"\n📄 Page {page_num + 1} – {len(images)} image(s)")

        for img_idx, img_info in enumerate(images):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                mime = base_image["ext"]
                print(f"   🖼️ Image {img_idx+1} ({len(img_bytes)} bytes, {mime})")
            except Exception as e:
                print(f"   ❌ Could not extract image {img_idx+1}: {e}")
                continue

            # Resize if necessary
            img_bytes = resize_image_bytes(img_bytes)

            # Send to Gemma
            print(f"   ⏳ Sending to Gemma...")
            description = describe_image_with_gemma(img_bytes)

            if description:
                print(f"   ✅ Gemma response:\n{description[:500]}{'...' if len(description) > 500 else ''}")
            else:
                print(f"   ⚠️ No description returned (API may have failed)")

            total_images += 1

    doc.close()
    print(f"\n🏁 Done. {total_images} image(s) processed.")

if __name__ == "__main__":
    main()