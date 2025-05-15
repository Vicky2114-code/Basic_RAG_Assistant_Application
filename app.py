import streamlit as st
from langchain.chains.llm import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ChatMessageHistory
from langchain_core.prompts import PromptTemplate

st.set_page_config(
    page_title="⚡ Smart PDF Chat (MongoDB)",
    layout="wide",
    page_icon="⚡"
)
import os
import hashlib
import time
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from sentence_transformers import util
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch

# ========== CONSTANTS ==========
PDF_UPLOAD_DIR = Path("uploaded_pdfs")
IRRELEVANT_THRESHOLD = 0.3  # Similarity threshold for relevance
TIMEOUT_SECONDS = 15  # Max time to wait for response
MAX_RETRIEVAL_DOCS = 3  # Number of docs to retrieve

# MongoDB Atlas Configuration
MONGO_URI = "mongodb+srv://vicky:vicky@cluster0.syoeipa.mongodb.net/RAG?retryWrites=true&w=majority"
DB_NAME = "RAG"
COLLECTION_NAME = "document_vectors"
INDEX_NAME = "vector_index_1"

# Response Templates
TEMPLATES = {
    "irrelevant": "I can only answer questions about the document. Please ask something related to the uploaded PDF.",
    "timeout": "I'm taking too long to respond. Please try a different question or simplify your query.",
    # "no_answer": "I couldn't find an answer in the document. Could you rephrase your question?",
    "welcome": "Upload a PDF document to get started!",
    "processing": "Processing your document...",
    "ready": "Document processed! Ask me anything about it.",
    "initial_think": "I'm getting ready to chat with you...",
    "db_error": "⚠️ Database connection failed. Please check your MongoDB connection settings.",
    "db_connected": "✅ Successfully connected to MongoDB Atlas"
}


# ========== DATABASE CONNECTION ==========
@st.cache_resource
def init_mongo_connection():
    """Initialize and cache the MongoDB connection"""
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


# Initialize MongoDB connection when app starts
mongo_client, mongo_error = init_mongo_connection()

if mongo_error:
    st.error(f"{TEMPLATES['db_error']}: {mongo_error}")
    st.stop()
else:
    st.toast(TEMPLATES["db_connected"], icon="✅")


# ========== CUSTOM HANDLERS ==========
class SmartStreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
        self.start_time = time.time()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text + "▌")

        # Timeout check
        if time.time() - self.start_time > TIMEOUT_SECONDS:
            raise TimeoutError(TEMPLATES["timeout"])


# ========== UTILITY FUNCTIONS ==========
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


def get_vectorstore(text: str, file_hash: str):
    """Create or load MongoDB Atlas vector store using Document objects"""
    try:
        collection = mongo_client[DB_NAME][COLLECTION_NAME]

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Check if document already exists
        existing_doc = collection.find_one({"metadata.document_hash": file_hash})
        if existing_doc:
            st.info("Loading existing knowledge base...")
            return MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=embeddings,
                index_name=INDEX_NAME,
                embedding_key="embedding",
                text_key="text",
                relevance_score_fn="cosine"
            )

        with st.spinner("Processing and indexing document..."):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                length_function=len
            )
            chunks = splitter.split_text(text)

            documents = [
                Document(page_content=chunk, metadata={"document_hash": file_hash})
                for chunk in chunks
            ]

            vectorstore = MongoDBAtlasVectorSearch.from_documents(
                documents=documents,
                embedding=embeddings,
                collection=collection,
                index_name=INDEX_NAME,
                embedding_key="embedding",
                text_key="text"
            )

        return vectorstore

    except Exception as e:
        st.error(f"Database operation failed: {str(e)}")
        st.stop()


def is_relevant(query: str, vectorstore) -> bool:
    """Check if query is relevant to document content"""
    try:
        if not vectorstore:
            return False
        print("vicky")
        # Get the most similar document
        docs = vectorstore.similarity_search(query, k=2)
        print("docs   ",docs)
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
        print(similarity)
        return similarity > IRRELEVANT_THRESHOLD
    except Exception as e:
        st.error(f"Relevance check error: {str(e)}")
        return False


def show_thinking_animation():
    """Show initial thinking animation when chat first loads"""
    with st.empty():
        for i in range(2):  # Only run for 2 cycles (about 1 second)
            dots = "." * (i % 4)
            st.markdown(f"🤔 {TEMPLATES['initial_think']}{dots}")
            time.sleep(0.5)


def clear_document_data(file_hash: str):
    """Remove all vectors for a specific document"""
    try:
        collection = mongo_client[DB_NAME][COLLECTION_NAME]
        result = collection.delete_many({"metadata.document_hash": file_hash})
        st.toast(f"Deleted {result.deleted_count} document chunks", icon="🗑️")
    except Exception as e:
        st.error(f"Failed to delete document: {str(e)}")


# ========== STREAMLIT APP ==========
def main():
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.initial_load = True

    if "processed_pdf_hash" not in st.session_state:
        st.session_state.processed_pdf_hash = None

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    PDF_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    st.title("⚡ Intelligent Document Assistant with MongoDB Atlas")

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox("Ollama Model", ["mistral", "llama3.2", "llama2"], index=0)
        temperature = st.slider("Response Creativity", 0.0, 1.0, 0.3, 0.1)

        if st.session_state.vectorstore and st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.initial_load = True
            st.rerun()

        if st.session_state.processed_pdf_hash and st.button("Remove Document from DB"):
            clear_document_data(st.session_state.processed_pdf_hash)
            st.session_state.processed_pdf_hash = None
            st.session_state.vectorstore = None
            st.session_state.chat_history = []
            st.success("Document removed from database!")
            st.rerun()

    # PDF upload
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    if uploaded_file:
        file_hash = get_file_hash(uploaded_file)
        uploaded_file.seek(0)

        if st.session_state.processed_pdf_hash != file_hash:
            with st.spinner(TEMPLATES["processing"]):
                st.session_state.processed_pdf_hash = file_hash
                st.session_state.chat_history = []
                st.session_state.initial_load = True

                pdf_path = save_uploaded_file(uploaded_file)
                raw_text = extract_text_from_pdf(uploaded_file)

                if raw_text:
                    st.session_state.vectorstore = get_vectorstore(raw_text, file_hash)
                    st.success(TEMPLATES["ready"])

    # Chat interface - optimized version
    if st.session_state.vectorstore:
        if st.session_state.initial_load:
            show_thinking_animation()
            st.session_state.initial_load = False

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Process user input
        if prompt := st.chat_input("Ask about the document"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Check question relevance first
                if not is_relevant(prompt, st.session_state.vectorstore):
                    st.markdown(TEMPLATES["irrelevant"])
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": TEMPLATES["irrelevant"]
                    })
                else:
                    stream_container = st.empty()
                    stream_handler = SmartStreamHandler(stream_container)

                    try:
                        # Simplified QA pipeline
                        llm = Ollama(
                            model=model_name,
                            temperature=0.5,
                            callbacks=[stream_handler]
                        )

                        # Configure retriever
                        retriever = st.session_state.vectorstore.as_retriever(
                            search_type="similarity",
                            search_kwargs={
                                "k": MAX_RETRIEVAL_DOCS,
                                "score_threshold": 0.4  # Only include highly relevant chunks
                            }
                        )

                        # Clean prompt template
                        qa_template = """Answer the question based on the document content below.
                        Provide a clear, structured response with:
                        1. A brief Summary Paragraph overview
                        2. 3-5 key points as bullet points
                        3. Any important details or numbers when relevant
                        4. also add like conclusion after complete the result
                        5. try to give depth answer 

                        Context: {context}
                        Question: {question}
                        """

                        # Create and run QA chain
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            return_source_documents=False,
                            # max_tokens_limit=1000,
                            chain_type_kwargs={
                                "prompt": PromptTemplate(
                                    template=qa_template,
                                    input_variables=["context", "question"]
                                )
                            }
                        )

                        response = qa_chain({"query": prompt})
                        answer = response["result"]

                        # Post-process answer formatting
                        answer = answer.replace("- ", "• ").replace("* ", "• ")  # Standardize bullets
                        stream_container.markdown(answer)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer
                        })

                    except Exception as e:
                        error_msg = f"⚠️ Error processing your request: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": TEMPLATES["no_answer"]
                        })
                        st.markdown(TEMPLATES["no_answer"])
    else:
        st.info(TEMPLATES["welcome"])
if __name__ == "__main__":
    main()