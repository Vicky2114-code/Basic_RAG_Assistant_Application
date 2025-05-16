from pydantic import BaseModel
from typing import Optional, List
from fastapi import UploadFile

class DocumentUpload(BaseModel):
    file: UploadFile
    model_name: Optional[str] = "mistral"
    temperature: Optional[float] = 0.3

class QueryRequest(BaseModel):
    question: str
    document_hash: str
    model_name: Optional[str] = "mistral"
    temperature: Optional[float] = 0.3

class QueryResponse(BaseModel):
    answer: str
    is_relevant: bool
    document_hash: str

class DocumentResponse(BaseModel):
    document_hash: str
    status: str
    message: str