import json

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pathlib import Path
from PyPDF2 import PdfReader
import hashlib
import io
import time
from pymongo import MongoClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_mongodb import MongoDBAtlasVectorSearch
import torch
from sentence_transformers import util

# ==== Constants ====
PDF_UPLOAD_DIR = Path("uploaded_pdfs")
MONGO_URI = "mongodb+srv://vicky:vicky@cluster0.syoeipa.mongodb.net/RAG?retryWrites=true&w=majority"
DB_NAME = "RAG"
COLLECTION_NAME = "document_vectors"
INDEX_NAME = "vector_index_1"
IRRELEVANT_THRESHOLD = 0.3

# ==== App & Init ====
app = FastAPI()
PDF_UPLOAD_DIR.mkdir(exist_ok=True)

# MongoDB
mongo_client = MongoClient(MONGO_URI)
collection = mongo_client[DB_NAME][COLLECTION_NAME]

# In-memory session state
vectorstore_cache = {}
file_hash_cache = {}


# ==== Utility ====
def get_file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())


def build_vectorstore(text: str, file_hash: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={'normalize_embeddings': True}
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"document_hash": file_hash}) for chunk in chunks]

    return MongoDBAtlasVectorSearch.from_documents(
        documents=documents,
        embedding=embeddings,
        collection=collection,
        index_name=INDEX_NAME,
        embedding_key="embedding",
        text_key="text"
    )


def is_relevant(query: str, vectorstore) -> bool:
    docs = vectorstore.similarity_search(query, k=2)
    if not docs:
        return False

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={'normalize_embeddings': True}
    )

    query_embedding = embeddings.embed_query(query)
    doc_embedding = embeddings.embed_query(docs[0].page_content)
    similarity = util.pytorch_cos_sim(torch.tensor(query_embedding), torch.tensor(doc_embedding)).item()
    return similarity > IRRELEVANT_THRESHOLD


# ==== Endpoints ====

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_bytes = await file.read()
    file_hash = get_file_hash(file_bytes)

    # Save file locally
    path = PDF_UPLOAD_DIR / file.filename
    path.write_bytes(file_bytes)

    # Check existing
    existing = collection.find_one({"metadata.document_hash": file_hash})
    if not existing:
        text = extract_text_from_pdf(file_bytes)
        vectorstore = build_vectorstore(text, file_hash)
    else:
        vectorstore = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            index_name=INDEX_NAME,
            embedding_key="embedding",
            text_key="text"
        )

    # Save to cache
    vectorstore_cache[file_hash] = vectorstore
    file_hash_cache["current"] = file_hash

    return {"status": "success", "file_hash": file_hash}


@app.post("/ask")
async def ask_question(request: Request):
    body = await request.json()
    question = body.get("question")
    file_hash = file_hash_cache.get("current")

    if not file_hash or file_hash not in vectorstore_cache:
        return JSONResponse({"error": "No document uploaded."}, status_code=400)

    vectorstore = vectorstore_cache[file_hash]

    if not is_relevant(question, vectorstore):
        return StreamingResponse(iter([
            json.dumps({"type": "error", "message": "I can only answer questions about the uploaded PDF."}) + "\n"
        ]), media_type="application/json")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    template = """Answer the question based on the document content below.
    1. Summary Paragraph
    2. 3-5 key points
    3. Important Details
    4. Conclusion

    Context: {context}
    Question: {question}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=Ollama(model="mistral", temperature=0.3),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )

    result = qa_chain({"query": question})
    answer = result["result"]

    # Chunk parsing: naive split by headers (for demo purposes)
    def parse_answer_sections(answer: str):
        chunks = {
            "summary": "",
            "key_points": [],
            "details": "",
            "conclusion": ""
        }

        lines = answer.split("\n")
        current_section = "summary"
        for line in lines:
            if "key points" in line.lower():
                current_section = "key_points"
            elif "details" in line.lower():
                current_section = "details"
            elif "conclusion" in line.lower():
                current_section = "conclusion"
            else:
                if current_section == "key_points" and ("•" in line or "-" in line):
                    chunks["key_points"].append(line.strip("•- "))
                else:
                    chunks[current_section] += line.strip() + " "

        return chunks

    structured_answer = parse_answer_sections(answer)

    def json_stream():
        for key, value in structured_answer.items():
            chunk = {"type": key, "content": value}
            yield json.dumps(chunk) + "\n"
            time.sleep(0.3)  # simulate delay for streaming effect

    return StreamingResponse(json_stream(), media_type="application/json")
