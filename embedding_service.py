import openai
from typing import List, Dict, Any
from config import config

class EmbeddingService:
    def __init__(self):
        openai.api_key = config.OPENAI_API_KEY
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Simple text chunking by character count
        """
        chunk_size = config.CHUNK_SIZE
        overlap = config.CHUNK_OVERLAP
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings using OpenAI's embedding model
        """
        try:
            response = openai.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            raise Exception(f"Failed to create embeddings: {str(e)}")
    
    def process_document(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a document: chunk it and create embeddings
        """
        # Chunk the text
        chunks = self.chunk_text(content)
        
        # Create embeddings for chunks
        embeddings = self.create_embeddings(chunks)
        
        # Prepare document data
        processed_data = {
            "chunks": chunks,
            "embeddings": embeddings,
            "metadata": metadata or {},
            "total_chunks": len(chunks)
        }
        
        return processed_data

# Initialize embedding service
embedding_service = EmbeddingService()