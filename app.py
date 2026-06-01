"""
Flask web server for PDF extraction.
Upload a PDF, choose Gemma on/off, and get extraction results.
"""

import os
import tempfile
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import json
import io

from core.ingestion.pdf_analyzer import analyze_pdf

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
UPLOAD_FOLDER = tempfile.gettempdir()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the frontend page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF upload, run extraction, return JSON results."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    # Get Gemma toggle value (default False)
    use_gemma = request.form.get('use_gemma', 'false').lower() == 'true'
    
    # Save uploaded file temporarily
    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(temp_path)
    
    try:
        # Run the extraction pipeline
        result = analyze_pdf(
            file_path=temp_path,
            use_gemma=use_gemma,
            domain="general"
        )
        
        # Return the result as JSON
        return jsonify({
            'success': True,
            'file_name': result['file_name'],
            'pdf_type': result['pdf_type'],
            'total_pages': result['total_pages'],
            'total_chunks': result['total_chunks'],
            'summary_text': result['summary_text'],
            'detailed_summary': result['detailed_summary'],
            'chunks': result['chunks']  # full chunks for download
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/download_json', methods=['POST'])
def download_json():
    """Download chunks as JSON file."""
    data = request.get_json()
    if not data or 'chunks' not in data:
        return jsonify({'error': 'No chunks data'}), 400
    
    file_name = data.get('file_name', 'extraction_result')
    json_str = json.dumps(data['chunks'], indent=2, ensure_ascii=False)
    
    return send_file(
        io.BytesIO(json_str.encode('utf-8')),
        mimetype='application/json',
        as_attachment=True,
        download_name=f"{file_name.replace('.pdf', '')}_chunks.json"
    )

if __name__ == '__main__':
    # Run on localhost:5000
    app.run(debug=True, host='0.0.0.0', port=5000)