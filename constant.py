from pathlib import Path

PDF_UPLOAD_DIR = Path("uploaded_pdfs")
IRRELEVANT_THRESHOLD = 0.3
TIMEOUT_SECONDS = 15
MAX_RETRIEVAL_DOCS = 3

TEMPLATES = {
    "irrelevant": "I can only answer questions about the document...",
    "timeout": "I'm taking too long to respond...",
    "welcome": "Upload a PDF document to get started!",
    "processing": "Processing your document...",
    "ready": "Document processed! Ask me anything...",
    "initial_think": "I'm getting ready to chat with you...",
    "db_error": "⚠️ Database connection failed...",
    "db_connected": "✅ Successfully connected to MongoDB Atlas"
}
