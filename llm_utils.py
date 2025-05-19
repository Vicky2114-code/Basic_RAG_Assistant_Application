import streamlit as st
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from sentence_transformers import util
from constant import IRRELEVANT_THRESHOLD

def get_vectorstore(text, file_hash, collection, index_name):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    existing_doc = collection.find_one({"document_hash": file_hash})
    if existing_doc:
        st.info("Loading existing knowledge base...")
        return MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embeddings,
    index_name=index_name,
    # embedding_key="embedding",
    # text_key="text"
)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"document_hash": file_hash}) for chunk in chunks]

    return MongoDBAtlasVectorSearch.from_documents(
                documents=documents,
                embedding=embeddings,
                collection=collection,
                index_name=index_name,
                embedding_key="embedding",
                text_key="text"
            )

def is_relevant(query, vectorstore) -> bool:
    docs = vectorstore.similarity_search(query, k=2)
    if not docs:
        return False

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    query_embedding = embeddings.embed_query(query)
    doc_embedding = embeddings.embed_query(docs[0].page_content)

    similarity = util.pytorch_cos_sim(query_embedding, doc_embedding).item()
    return similarity > IRRELEVANT_THRESHOLD
