# extract_text.py
import os
from pdfminer.high_level import extract_text
from docx import Document
from PIL import Image
import pytesseract
from config import DIRECTORIES, TESSERACT_PATH

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_text_from_file(file_path):
    """Extract text from PDFs, Word documents, and images."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_text(file_path)
    elif ext in [".docx", ".doc"]:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif ext in [".png", ".jpg", ".jpeg"]:
        return pytesseract.image_to_string(Image.open(file_path))
    return None  # Unsupported file type

def extract_text_from_all_files():
    """Iterates over all files and extracts text."""
    extracted_data = []

    for category, dir_path in DIRECTORIES.items():
        for root, _, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                text = extract_text_from_file(file_path)

                if text:
                    extracted_data.append({
                        "file_name": file,
                        "text": text[:1000]  # Limit text for efficiency
                    })

    return extracted_data
