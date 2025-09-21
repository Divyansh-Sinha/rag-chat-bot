from pydantic import BaseModel
from typing import List, Optional

class DocumentUpload(BaseModel):
    content: str
    metadata: Optional[dict] = {}

class QueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: Optional[float] = None

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None