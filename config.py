from pathlib import Path

# Directories
PDF_UPLOAD_DIR = Path("uploaded_pdfs")
PDF_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# MongoDB Configuration
MONGO_URI = "mongodb+srv://vicky:vicky@cluster0.syoeipa.mongodb.net/RAG?retryWrites=true&w=majority"
DB_NAME = "RAG"
COLLECTION_NAME = "document_vectors"
INDEX_NAME = "vector_index_1"

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "mistral"

# Processing Parameters
IRRELEVANT_THRESHOLD = 0.3  # Similarity threshold for relevance
MAX_RETRIEVAL_DOCS = 3      # Number of docs to retrieve
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Response Templates
TEMPLATES = {
    "irrelevant": "I can only answer questions about the document. Please ask something related to the uploaded PDF.",
    "timeout": "I'm taking too long to respond. Please try a different question or simplify your query.",
    "welcome": "Upload a PDF document to get started!",
    "processing": "Processing your document...",
    "ready": "Document processed! Ask me anything about it.",
    "db_error": "⚠️ Database connection failed. Please check your MongoDB connection settings.",
    "db_connected": "✅ Successfully connected to MongoDB Atlas"
}