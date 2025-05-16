import hashlib
from pathlib import Path
from PyPDF2 import PdfReader
import streamlit as st
from constant import PDF_UPLOAD_DIR

def get_file_hash(file) -> str:
    file.seek(0)
    return hashlib.md5(file.read()).hexdigest()

def extract_text_from_pdf(file) -> str:
    try:
        pdf = PdfReader(file)
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def save_uploaded_file(uploaded_file) -> Path:
    file_path = PDF_UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path
