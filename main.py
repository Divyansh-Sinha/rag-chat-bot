from fastapi import FastAPI, HTTPException, File, UploadFile, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
import uvicorn
from typing import List, Optional
import json
import time

from logging_config import logger

from models import DocumentUpload, QueryRequest, QueryResponse, APIResponse, UserCreate, UserLogin
from embedding_service import embedding_service
from vector_store import vector_store
from query_service import rag_orchestrator
from file_processor import file_processor
from firebase_admin_auth import verify_firebase_token, generate_api_key, validate_api_key, create_firebase_user, login_with_email_and_password

# Initialize FastAPI app
app = FastAPI(
    title="Simple RAG API",
    description="A simple Retrieval-Augmented Generation API using FastAPI, FAISS, and OpenAI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.security import HTTPBearer, APIKeyHeader

# Security schemes
http_bearer_scheme = HTTPBearer()
api_key_scheme = APIKeyHeader(name='X-API-Key')

async def get_current_user(token: str = Depends(http_bearer_scheme)) -> str:
    user_id = verify_firebase_token(token.credentials)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user_id

async def api_key_header(x_api_key: str = Depends(api_key_scheme)):
    if not validate_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or expired API Key")
    return x_api_key

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Log incoming requests and their processing time.
    """
    start_time = time.time()
    logger.info(f"Request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - Process time: {process_time:.4f}s")
    
    return response

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG API is running", "status": "healthy"}

@app.post("/register")
async def register_user(user_data: UserCreate):
    """
    Register a new user.
    """
    user_id = create_firebase_user(user_data.email, user_data.password)
    if not user_id:
        raise HTTPException(status_code=400, detail="Could not create user. The email might already be in use.")
    
    return APIResponse(success=True, message="User created successfully.")

@app.post("/login")
async def login_for_id_token(user_data: UserLogin):
    """
    Login a user and return an ID token.
    """
    user_id, id_token = login_with_email_and_password(user_data.email, user_data.password)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    return APIResponse(success=True, message="Login successful", data={"id_token": id_token})


@app.post("/generate-key")
async def generate_new_api_key(user_id: str = Depends(get_current_user), name: Optional[str] = None):
    """Generate a new API key for the authenticated user."""
    api_key = generate_api_key(user_id, name)
    return APIResponse(success=True, message="API Key generated successfully", data={"api_key": api_key, "name": name})

@app.get("/api-keys")
async def get_user_api_keys_endpoint(user_id: str = Depends(get_current_user)):
    """Retrieve all API keys for the authenticated user."""
    from firebase_admin_auth import get_user_api_keys
    try:
        user_keys = get_user_api_keys(user_id)
        return APIResponse(
            success=True, 
            message="API keys retrieved successfully", 
            data={"api_keys": user_keys}
        )
    except Exception as e:
        logger.error(f"Error retrieving API keys: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve API keys: {str(e)}")

@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats"""
    return {
        "supported_formats": file_processor.get_supported_formats(),
        "descriptions": {
            ".txt": "Plain text files",
            ".pdf": "PDF documents",
            ".xlsx": "Excel spreadsheets (2007+)",
            ".xls": "Excel spreadsheets (legacy)",
            ".docx": "Word documents (2007+)",
            ".doc": "Word documents (legacy)"
        }
    }

@app.get("/stats", dependencies=[Depends(api_key_header)])
async def get_stats():
    """Get vector database statistics"""
    try:
        stats = vector_store.get_stats()
        return APIResponse(
            success=True,
            message="Statistics retrieved successfully",
            data=stats
        )
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/text", dependencies=[Depends(api_key_header)])
async def upload_text_document(document: DocumentUpload):
    """
    Upload and process text document for embedding and storage
    """
    try:
        # Process the document
        processed_data = embedding_service.process_document(
            content=document.content,
            metadata=document.metadata
        )
        
        # Add to vector store
        success = vector_store.add_documents(processed_data)
        
        if success:
            return APIResponse(
                success=True,
                message=f"Document processed successfully. Created {processed_data['total_chunks']} chunks.",
                data={"chunks_created": processed_data['total_chunks']}
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to store document")
            
    except Exception as e:
        logger.error(f"Error processing text document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.post("/upload/file", dependencies=[Depends(api_key_header)])
async def upload_file_document(file: UploadFile = File(...)):
    """Upload and process various file formats (PDF, Excel, Word, Text)"""
    try:
        # Read file content
        file_content = await file.read()
        
        # Process file based on its format
        processing_result = file_processor.process_file(file_content, file.filename)
        
        if not processing_result['success']:
            raise HTTPException(status_code=400, detail=processing_result['error'])
        
        # Process the extracted content
        processed_data = embedding_service.process_document(
            content=processing_result['content'],
            metadata=processing_result['metadata']
        )
        
        # Add to vector store
        success = vector_store.add_documents(processed_data)
        
        if success:
            return APIResponse(
                success=True,
                message=f"File '{file.filename}' processed successfully. Created {processed_data['total_chunks']} chunks.",
                data={
                    "filename": file.filename,
                    "file_type": processing_result['metadata']['file_type'],
                    "chunks_created": processed_data['total_chunks'],
                    "content_length": processing_result['metadata']['content_length']
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to store document")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.post("/query", response_model=QueryResponse, dependencies=[Depends(api_key_header)])
async def query_documents(request: QueryRequest):
    """
    Query the knowledge base and get AI-generated response
    """
    try:
        # Process query through RAG pipeline
        result = rag_orchestrator.process_query(request.query)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"][:request.max_results],
            confidence=1.0 if result["context_used"] else 0.0
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.delete("/clear", dependencies=[Depends(api_key_header)])
async def clear_database():
    """
    Clear all documents from the vector database
    """
    try:
        # Recreate the vector store (this clears all data)
        vector_store._create_index()
        vector_store._save_index()
        
        return APIResponse(
            success=True,
            message="Vector database cleared successfully"
        )
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)