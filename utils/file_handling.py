import hashlib
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import Tuple, Optional,List


async def get_file_hash(file) -> str:
    """Generate MD5 hash of file content"""
    print(file)
    await file.seek(0)  # ensure you're at the beginning
    contents = await file.read()
    print(contents)
    file_hash = hashlib.md5(contents).hexdigest()
    print(file_hash)
    print("File hash:", file_hash)
    return file_hash
def extract_text_from_pdf(file) -> Tuple[str, Optional[str]]:
    """Extract text from PDF file"""
    try:
        pdf = PdfReader(file)
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        return text, None
    except Exception as e:
        return None, str(e)

def save_uploaded_file(uploaded_file, upload_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Save uploaded file to disk"""
    try:
        file_path = upload_dir / uploaded_file.filename
        with open(file_path, "wb") as f:
            f.write(uploaded_file.file.read())
        return file_path, None
    except Exception as e:
        return None, str(e)

def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)