from pymongo import MongoClient
from config import MONGO_URI, DB_NAME, COLLECTION_NAME, TEMPLATES,INDEX_NAME
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Tuple, Optional


def get_mongo_client() -> Tuple[Optional[MongoClient], Optional[str]]:
    """Initialize and return MongoDB client"""
    try:
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000
        )
        client.admin.command('ping')  # Check connection
        return client, None
    except Exception as e:
        return None, str(e)


def get_vectorstore(collection, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Get MongoDB Atlas vector store"""
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=INDEX_NAME,
        embedding_key="embedding",
        text_key="text",
        relevance_score_fn="cosine"
    )