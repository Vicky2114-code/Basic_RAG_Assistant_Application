from sentence_transformers import util
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from config import IRRELEVANT_THRESHOLD


def is_relevant(query: str, vectorstore, threshold: float = IRRELEVANT_THRESHOLD) -> bool:
    """Check if query is relevant to document content"""
    if not vectorstore:
        return False

    # Get the most similar document
    docs = vectorstore.similarity_search(query, k=2)
    if not docs:
        return False

    # Calculate semantic similarity
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    query_embedding = embeddings.embed_query(query)
    doc_embedding = embeddings.embed_query(docs[0].page_content)

    similarity = util.pytorch_cos_sim(query_embedding, doc_embedding).item()
    return similarity > threshold