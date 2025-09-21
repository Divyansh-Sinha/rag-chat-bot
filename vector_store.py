import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple
from config import config

class FAISSVectorStore:
    def __init__(self):
        self.index = None
        self.documents = []  # Store document chunks and metadata
        self.dimension = 1536  # OpenAI embedding dimension
        self.index_path = config.VECTOR_DB_PATH
        
        # Create directory if it doesn't exist
        os.makedirs(self.index_path, exist_ok=True)
        
        # Try to load existing index
        self._load_index()
    
    def _create_index(self):
        """Create a new FAISS index"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []
    
    def _save_index(self):
        """Save FAISS index and document metadata"""
        try:
            if self.index is not None:
                # Save FAISS index
                faiss.write_index(self.index, os.path.join(self.index_path, "faiss.index"))
                
                # Save document metadata
                with open(os.path.join(self.index_path, "documents.pkl"), "wb") as f:
                    pickle.dump(self.documents, f)
                    
                print(f"Index saved with {len(self.documents)} documents")
        except Exception as e:
            raise Exception(f"Failed to save index: {str(e)}")
    
    def _load_index(self):
        """Load existing FAISS index and document metadata"""
        index_file = os.path.join(self.index_path, "faiss.index")
        docs_file = os.path.join(self.index_path, "documents.pkl")
        
        if os.path.exists(index_file) and os.path.exists(docs_file):
            try:
                # Load FAISS index
                self.index = faiss.read_index(index_file)
                
                # Load document metadata
                with open(docs_file, "rb") as f:
                    self.documents = pickle.load(f)
                    
                print(f"Loaded existing index with {len(self.documents)} documents")
            except Exception as e:
                print(f"Failed to load existing index: {str(e)}")
                self._create_index()
        else:
            self._create_index()
    
    def add_documents(self, processed_data: Dict[str, Any]) -> bool:
        """Add processed document data to the vector store"""
        try:
            embeddings = processed_data["embeddings"]
            chunks = processed_data["chunks"]
            metadata = processed_data["metadata"]
            
            # Convert embeddings to numpy array
            embedding_matrix = np.array(embeddings, dtype=np.float32)
            
            # Add embeddings to FAISS index
            self.index.add(embedding_matrix)
            
            # Store document chunks with metadata
            for i, chunk in enumerate(chunks):
                doc_data = {
                    "chunk": chunk,
                    "metadata": metadata,
                    "chunk_index": i,
                    "doc_id": len(self.documents)
                }
                self.documents.append(doc_data)
            
            # Save the updated index
            self._save_index()
            
            return True
        except Exception as e:
            raise Exception(f"Failed to add documents: {str(e)}")
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        try:
            # Convert query embedding to numpy array with proper shape
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Ensure k doesn't exceed available documents
            k = min(k, len(self.documents))
            
            # Search in FAISS index
            distances, indices = self.index.search(query_vector, k)
            
            # Retrieve matching documents
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.documents):  # Check for valid index
                    doc = self.documents[idx].copy()
                    doc["similarity_score"] = float(distances[0][i])
                    results.append(doc)
            
            return results
        except Exception as e:
            raise Exception(f"Failed to search: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension
        }

# Initialize vector store
vector_store = FAISSVectorStore()