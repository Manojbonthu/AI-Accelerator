import sys, os, json
sys.path.insert(0, '.')
from core.ingestion.pdf_analyzer import analyze_pdf

filepath = "tests/sample_docs/c4611_sample_explain.pdf"
result = analyze_pdf(filepath, use_gemma=False)   # set use_gemma=True if you want descriptions

# Print the first chunk as JSON to check the format
if result['chunks']:
    print(json.dumps(result['chunks'][0], indent=2))
else:
    print("No chunks created.")