from queue import Queue
from threading import Thread

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import StreamingResponse, JSONResponse
import hashlib
import io
import json
import os
from langchain_core.callbacks import CallbackManager, BaseCallbackHandler

from pdf_utils import extract_text_from_pdf
from config import DB_NAME, COLLECTION_NAME, INDEX_NAME
from db import init_mongo_connection
from llm_utils import get_vectorstore
from langchain_community.llms import Ollama
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from starlette.concurrency import run_in_threadpool
from ollama import Client


app = FastAPI()



# Load model and base URL from env
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://0.0.0.0:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")

# Ollama client instance
ollama_client = Client(host=OLLAMA_BASE_URL)

# Token generator
def generate_tokens(question: str):
    for chunk in ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": question}],
        stream=True
    ):
        yield chunk["message"]["content"]

# Generator for StreamingResponse
def generate_json_stream(question: str):
    full_content = ""
    for token in generate_tokens(question):
        full_content += token
        yield json.dumps({
            "model": OLLAMA_MODEL,
            "content": token,
            "done": False
        }).encode('utf-8') + b'\n'

    # Final full chunk
    yield json.dumps({
        "model": OLLAMA_MODEL,
        "full_content": full_content,
        "done": True
    }).encode('utf-8') + b'\n'

# FastAPI route
@app.post("/users/chat")
async def ask_ai(request: Request):
    body = await request.json()
    question = body.get("question")

    if not question:
        return {"error": "Missing 'question'"}

    # Run token generation in thread-safe context
    return StreamingResponse(
        generate_json_stream(question),
        media_type="application/json"
    )
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

    # DB setup (replace with your actual logic)
    client, err = init_mongo_connection()
    if err:
        return JSONResponse({"status": "error", "message": str(err)}, status_code=500)

    collection = client[DB_NAME][COLLECTION_NAME]
    vectorstore = get_vectorstore(None, file_hash, collection, INDEX_NAME)

    if not vectorstore:
        return JSONResponse({"status": "error", "message": "No vectorstore found"}, status_code=404)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5, "score_threshold": 0.4}
    )

    # Token queue and full content tracker
    token_queue = Queue()
    full_content = []

    class StreamingHandler(BaseCallbackHandler):
        def on_llm_new_token(self, token: str, **kwargs):
            token_queue.put(token)
            full_content.append(token)

    handler = StreamingHandler()
    callback_manager = CallbackManager([handler])

    llm = Ollama(
        model=model_name,
        temperature=temperature,
        callback_manager=callback_manager
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template}
    )

    async def final_chunk_stream():
        yield json.dumps({"event": "start", "message": "Running QA..."}) + "\n"

        def run_chain():
            try:
                qa_chain({"query": prompt})
            except Exception as e:
                token_queue.put(e)
            token_queue.put(None)

        Thread(target=run_chain).start()

        while True:
            token = token_queue.get()
            if token is None:
                break
            elif isinstance(token, Exception):
                yield json.dumps({"event": "error", "message": str(token)}) + "\n"
                return
            else:
                yield json.dumps({"event": "token", "token": token}) + "\n"

        full_answer = "".join(full_content)
        yield json.dumps({"event": "answer", "data": full_answer}) + "\n"
        yield json.dumps({"event": "end", "message": "Completed"}) + "\n"

    return StreamingResponse(final_chunk_stream(), media_type="application/x-ndjson")