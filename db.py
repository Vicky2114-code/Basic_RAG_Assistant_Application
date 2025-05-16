import streamlit as st
from pymongo import MongoClient
from config import MONGO_URI
from constant import TEMPLATES

@st.cache_resource
def init_mongo_connection():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        return client, None
    except Exception as e:
        return None, str(e)

def clear_document_data(client, db_name, collection_name, file_hash: str):
    collection = client[db_name][collection_name]
    result = collection.delete_many({"metadata.document_hash": file_hash})
    st.toast(f"Deleted {result.deleted_count} document chunks", icon="🗑️")
