import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple
from config import config
from logging_config import logger
import shutil

class FAISSVectorStore:
    def __init__(self):
        self.dimension = 1536  # OpenAI embedding dimension
        self.base_path = config.VECTOR_DB_PATH
        self.indexes = {}  # Cache for user indexes: {user_id: (index, documents)}
        
        # Create base directory if it doesn't exist
        os.makedirs(self.base_path, exist_ok=True)

    def _get_user_index_path(self, user_id: str) -> str:
        return os.path.join(self.base_path, user_id)

    def _load_index(self, user_id: str):
        """Load or create an index for a specific user."""
        if user_id in self.indexes:
            return self.indexes[user_id]

        user_index_path = self._get_user_index_path(user_id)
        index_file = os.path.join(user_index_path, "faiss.index")
        docs_file = os.path.join(user_index_path, "documents.pkl")

        if os.path.exists(index_file) and os.path.exists(docs_file):
            try:
                logger.info(f"Loading existing index for user {user_id}...")
                index = faiss.read_index(index_file)
                with open(docs_file, "rb") as f:
                    documents = pickle.load(f)
                logger.info(f"Loaded existing index for user {user_id} with {len(documents)} documents")
                self.indexes[user_id] = (index, documents)
                return index, documents
            except Exception as e:
                logger.error(f"Failed to load existing index for user {user_id}: {str(e)}", exc_info=True)
                # If loading fails, create a new one
                return self._create_index(user_id)
        else:
            logger.info(f"No existing index found for user {user_id}. Creating a new one.")
            return self._create_index(user_id)

    def _create_index(self, user_id: str) -> Tuple[faiss.Index, List]:
        """Create a new FAISS index for a user."""
        user_index_path = self._get_user_index_path(user_id)
        os.makedirs(user_index_path, exist_ok=True)
        
        logger.info(f"Creating new FAISS index for user {user_id}.")
        index = faiss.IndexFlatL2(self.dimension)
        documents = []
        self.indexes[user_id] = (index, documents)
        return index, documents

    def _save_index(self, user_id: str):
        """Save FAISS index and document metadata for a user."""
        if user_id not in self.indexes:
            logger.warning(f"Attempted to save index for user {user_id}, but it's not loaded.")
            return

        try:
            index, documents = self.indexes[user_id]
            user_index_path = self._get_user_index_path(user_id)
            
            if index is not None:
                faiss.write_index(index, os.path.join(user_index_path, "faiss.index"))
                with open(os.path.join(user_index_path, "documents.pkl"), "wb") as f:
                    pickle.dump(documents, f)
                logger.info(f"Index for user {user_id} saved with {len(documents)} documents")
        except Exception as e:
            logger.error(f"Failed to save index for user {user_id}: {str(e)}", exc_info=True)
            raise Exception(f"Failed to save index for user {user_id}: {str(e)}")

    def add_documents(self, user_id: str, processed_data: Dict[str, Any]) -> bool:
        """Add processed document data to a user's vector store."""
        logger.info(f"Adding {processed_data['total_chunks']} new chunks to the vector store for user {user_id}.")
        try:
            index, documents = self._load_index(user_id)
            
            embeddings = processed_data["embeddings"]
            chunks = processed_data["chunks"]
            metadata = processed_data["metadata"]
            
            embedding_matrix = np.array(embeddings, dtype=np.float32)
            index.add(embedding_matrix)
            
            for i, chunk in enumerate(chunks):
                doc_data = {
                    "chunk": chunk,
                    "metadata": metadata,
                    "chunk_index": i,
                    "doc_id": len(documents)
                }
                documents.append(doc_data)
            
            self._save_index(user_id)
            
            logger.info(f"Documents added successfully for user {user_id}.")
            return True
        except Exception as e:
            logger.error(f"Failed to add documents for user {user_id}: {str(e)}", exc_info=True)
            raise Exception(f"Failed to add documents for user {user_id}: {str(e)}")

    def search(self, user_id: str, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents in a user's index."""
        index, documents = self._load_index(user_id)
        
        if index is None or len(documents) == 0:
            logger.warning(f"Search attempted on an empty vector store for user {user_id}.")
            return []
        
        logger.info(f"Searching for {k} nearest neighbors for user {user_id}.")
        try:
            query_vector = np.array([query_embedding], dtype=np.float32)
            k = min(k, len(documents))
            
            distances, indices = index.search(query_vector, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(documents):
                    doc = documents[idx].copy()
                    doc["similarity_score"] = float(distances[0][i])
                    results.append(doc)
            
            logger.info(f"Found {len(results)} matching documents for user {user_id}.")
            return results
        except Exception as e:
            logger.error(f"Failed to search for user {user_id}: {str(e)}", exc_info=True)
            raise Exception(f"Failed to search for user {user_id}: {str(e)}")

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Get vector store statistics for a user."""
        index, documents = self._load_index(user_id)
        return {
            "user_id": user_id,
            "total_documents": len(documents),
            "index_size": index.ntotal if index else 0,
            "dimension": self.dimension
        }

    def clear_user_index(self, user_id: str) -> bool:
        """
        Clear all documents from a user's vector database by deleting their data directory.
        Returns True if the directory was deleted, False otherwise.
        """
        user_index_path = self._get_user_index_path(user_id)
        if user_id in self.indexes:
            del self.indexes[user_id]
        
        if os.path.exists(user_index_path):
            shutil.rmtree(user_index_path)
            logger.info(f"Deleted user data directory for user {user_id}")
            return True
        
        logger.info(f"No data directory found for user {user_id}. Nothing to clear.")
        return False

# Initialize vector store
vector_store = FAISSVectorStore()
