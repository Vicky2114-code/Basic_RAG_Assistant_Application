from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import StreamingResponse, JSONResponse
import hashlib
import io
import json

from pdf_utils import extract_text_from_pdf
from config import DB_NAME, COLLECTION_NAME, INDEX_NAME
from db import init_mongo_connection
from llm_utils import get_vectorstore
from langchain_community.llms import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

app = FastAPI()

def calculate_file_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_bytes = await file.read()
    file_hash = calculate_file_hash(file_bytes)
    raw_text = extract_text_from_pdf(io.BytesIO(file_bytes))

    if not raw_text:
        return JSONResponse({"status": "error", "message": "Empty or invalid PDF"}, status_code=400)

    client, err = init_mongo_connection()
    if err:
        return JSONResponse({"status": "error", "message": str(err)}, status_code=500)

    collection = client[DB_NAME][COLLECTION_NAME]
    get_vectorstore(raw_text, file_hash, collection, INDEX_NAME)

    return {"status": "success", "hash": file_hash}

prompt_template = PromptTemplate(
    template="""Answer the question based on the document content below.
1. Summary Paragraph  
2. 3–5 bullet points  
3. Important details/numbers  
4. Conclusion  
5. Provide depth

Context: {context}
Question: {question}""",
    input_variables=["context", "question"]
)

@app.post("/chat/{file_hash}")
async def chat_with_pdf(file_hash: str, request: Request):
    body = await request.json()
    prompt = body.get("prompt")
    model_name = body.get("model", "deepseek-r1:1.5b")
    temperature = body.get("temperature", 0.3)

    client, err = init_mongo_connection()
    if err:
        return JSONResponse({"status": "error", "message": str(err)}, status_code=500)

    collection = client[DB_NAME][COLLECTION_NAME]
    vectorstore = get_vectorstore(None, file_hash, collection, INDEX_NAME)

    if not vectorstore:
        return JSONResponse({"status": "error", "message": "No vectorstore found"}, status_code=404)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5, "score_threshold": 0.4})
    llm = Ollama(model=model_name, temperature=temperature)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

    def generate():
        # First event (streaming start)
        yield json.dumps({"status": "streaming", "answer": ""}) + "\n"

        # Simulate processing
        result = qa_chain({"query": prompt})
        answer = result["result"].strip()
        yield json.dumps({"status": "complete", "answer": answer}) + "\n"

    return StreamingResponse(generate(), media_type="application/json")
