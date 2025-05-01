import streamlit as st
import os
import hashlib
import time
from pathlib import Path
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from sentence_transformers import util

# ========== CONSTANTS ==========
VECTOR_STORE_DIR = Path("knowledge_base/faiss_index")
PDF_UPLOAD_DIR = Path("uploaded_pdfs")
IRRELEVANT_THRESHOLD = 0.7  # Similarity threshold for relevance
TIMEOUT_SECONDS = 15  # Max time to wait for response
MAX_RETRIEVAL_DOCS = 3  # Number of docs to retrieve

# Response Templates
TEMPLATES = {
    "irrelevant": "I can only answer questions about the document. Please ask something related to the uploaded PDF.",
    "timeout": "I'm taking too long to respond. Please try a different question or simplify your query.",
    "no_answer": "I couldn't find an answer in the document. Could you rephrase your question?",
    "welcome": "Upload a PDF document to get started!",
    "processing": "Processing your document...",
    "ready": "Document processed! Ask me anything about it.",
    "initial_think": "I'm getting ready to chat with you..."
}


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


def get_vectorstore(text, file_hash: str):
    pdf_vectorstore_path = VECTOR_STORE_DIR / f"{file_hash}_index"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    if pdf_vectorstore_path.exists():
        st.info("Loading existing knowledge base...")
        return FAISS.load_local(
            str(pdf_vectorstore_path),
            embeddings,
            allow_dangerous_deserialization=True
        )

    with st.spinner(TEMPLATES["processing"]):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunks = splitter.split_text(text)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        vectorstore.save_local(str(pdf_vectorstore_path))
    return vectorstore


def is_relevant(query: str, vectorstore) -> bool:
    """Check if query is relevant to document content"""
    try:
        if not vectorstore:
            return False

        # Get most similar chunk
        docs = vectorstore.similarity_search(query, k=1)
        if not docs:
            return False

        # Calculate semantic similarity
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        query_embedding = embeddings.embed_query(query)
        doc_embedding = embeddings.embed_query(docs[0].page_content)

        similarity = util.pytorch_cos_sim(query_embedding, doc_embedding).item()
        return similarity > IRRELEVANT_THRESHOLD
    except:
        return False


def show_thinking_animation():
    """Show initial thinking animation when chat first loads"""
    with st.empty():
        for i in range(2):  # Only run for 2 cycles (about 1 second)
            dots = "." * (i % 4)
            st.markdown(f"🤔 {TEMPLATES['initial_think']}{dots}")
            time.sleep(0.5)


# ========== STREAMLIT APP ==========
def main():
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        st.session_state.initial_load = True  # New flag for initial load

    if "processed_pdf_hash" not in st.session_state:
        st.session_state.processed_pdf_hash = None

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Create directories
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # UI Configuration
    st.set_page_config(
        page_title="⚡ Smart PDF Chat",
        layout="wide",
        page_icon="⚡"
    )
    st.title("⚡ Intelligent Document Assistant using sentence-transformers/all-MiniLM-L6-v2 and ollama LLM mistral ")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        model_name = st.selectbox(
            "Ollama Model",
            ["mistral", "llama3.2", "llama2"],
            index=0
        )
        temperature = st.slider(
            "Response Creativity",
            0.0, 1.0, 0.3, 0.1
        )

        if st.session_state.vectorstore and st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.initial_load = True  # Reset on clear
            st.rerun()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"]
    )

    # Process uploaded file
    if uploaded_file:
        file_hash = get_file_hash(uploaded_file)
        uploaded_file.seek(0)

        if st.session_state.processed_pdf_hash != file_hash:
            with st.spinner(TEMPLATES["processing"]):
                st.session_state.processed_pdf_hash = file_hash
                st.session_state.chat_history = []
                st.session_state.initial_load = True  # Reset on new doc

                pdf_path = save_uploaded_file(uploaded_file)
                raw_text = extract_text_from_pdf(uploaded_file)

                if raw_text:
                    st.session_state.vectorstore = get_vectorstore(raw_text, file_hash)
                    st.success(TEMPLATES["ready"])

    # Chat interface
    if st.session_state.vectorstore:
        # Show initial thinking animation only once
        if st.session_state.initial_load:
            show_thinking_animation()
            st.session_state.initial_load = False

        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": MAX_RETRIEVAL_DOCS,
                "fetch_k": min(10, MAX_RETRIEVAL_DOCS * 3)
            }
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about the document"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Immediate relevance check
                if not is_relevant(prompt, st.session_state.vectorstore):
                    st.markdown(TEMPLATES["irrelevant"])
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": TEMPLATES["irrelevant"]
                    })
                else:
                    # Prepare for fast response
                    stream_container = st.empty()
                    stream_handler = SmartStreamHandler(stream_container)

                    try:
                        llm = Ollama(
                            model=model_name,
                            temperature=temperature,
                            callbacks=[stream_handler]
                        )

                        qa_chain = ConversationalRetrievalChain.from_llm(
                            llm=llm,
                            retriever=retriever,
                            memory=memory,
                            return_source_documents=False,
                            max_tokens_limit=500,
                            verbose=False
                        )

                        # Get response with timeout protection
                        response = qa_chain({"question": prompt})

                        # Finalize response
                        stream_container.markdown(stream_handler.text)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": stream_handler.text
                        })

                    except TimeoutError:
                        st.markdown(TEMPLATES["timeout"])
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": TEMPLATES["timeout"]
                        })
                    except Exception:
                        st.markdown(TEMPLATES["no_answer"])
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": TEMPLATES["no_answer"]
                        })
    else:
        st.info(TEMPLATES["welcome"])


if __name__ == "__main__":
    main()