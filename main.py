from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List
import json

from models import DocumentUpload, QueryRequest, QueryResponse, APIResponse
from embedding_service import embedding_service
from vector_store import vector_store
from query_service import rag_orchestrator
from file_processor import file_processor  # Import the new file processor

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

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG API is running", "status": "healthy"}

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

@app.get("/stats")
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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/text")
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
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.post("/upload/file")
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
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
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
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.delete("/clear")
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
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)