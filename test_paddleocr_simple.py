from paddleocr import PaddleOCR
import fitz

# Initialize OCR (old API – no enable_mkldnn needed)
ocr = PaddleOCR(lang='en', use_angle_cls=True)

# Test on the first page of your PDF
doc = fitz.open("tests/sample_docs/PublicWaterMassMailing.pdf")
page = doc[0]
pix = page.get_pixmap(dpi=150)
img = pix.tobytes("png")

# Run OCR (batch mode: list of images)
result = ocr.ocr(img)
print("OCR succeeded, number of text lines found:", len(result[0]) if result else 0)
doc.close()