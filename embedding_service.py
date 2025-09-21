import openai
from typing import List, Dict, Any
from config import config
from logging_config import logger

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
            logger.info(f"Creating embeddings for {len(texts)} chunks.")
            response = openai.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=texts
            )
            logger.info("Embeddings created successfully.")
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f"Failed to create embeddings: {str(e)}", exc_info=True)
            raise Exception(f"Failed to create embeddings: {str(e)}")
    
    def process_document(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a document: chunk it and create embeddings
        """
        logger.info("Processing document...")
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
        
        logger.info(f"Document processed. Total chunks: {len(chunks)}")
        return processed_data

# Initialize embedding service
embedding_service = EmbeddingService()