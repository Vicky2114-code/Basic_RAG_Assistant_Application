from typing import Tuple, Optional
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
from utils.database import get_mongo_client, get_vectorstore
from utils.file_handling import get_file_hash, extract_text_from_pdf, split_text
from config import DB_NAME, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, TEMPLATES,INDEX_NAME


class DocumentService:
    def __init__(self):
        self.client, self.error = get_mongo_client()
        if self.error:
            raise ConnectionError(f"{TEMPLATES['db_error']}: {self.error}")

        self.collection = self.client[DB_NAME][COLLECTION_NAME]
        self.vectorstore = get_vectorstore(self.collection)

    def process_document(self, file) -> Tuple[Optional[str], Optional[str]]:
        """Process uploaded PDF document"""
        try:
            # Get file hash and check if already processed
            file_hash = get_file_hash(file)
            print("file_hash   ",file_hash)
            existing_doc = self.collection.find_one({"metadata.document_hash": file_hash})

            if existing_doc:
                return file_hash, "Document already processed"
            print("vicky222")
            # Extract and split text
            text, error = extract_text_from_pdf(file)
            print(text)
            if error:
                return None, error

            chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            documents = [
                Document(page_content=chunk, metadata={"document_hash": file_hash})
                for chunk in chunks
            ]

            # Store in vector database
            self.vectorstore.from_documents(
                documents=documents,
                embedding=self.vectorstore.embedding,
                collection=self.collection,
                index_name=INDEX_NAME,
                embedding_key="embedding",
                text_key="text"
            )

            return file_hash, None
        except Exception as e:
            return None, str(e)