# RAG API Application Flow

This document outlines the application flow for the multi-tenant RAG API, detailing user authentication, data ingestion, and querying processes.

## 1. Authentication Flow

The authentication is handled by Firebase. Users can register, log in to get an ID token, and then use this token to generate API keys for accessing the RAG functionalities.

### 1.1. User Registration and Login

- **Registration**: New users are created using an email and password.
- **Login**: Registered users can log in to receive a Firebase ID token.

*Relevant Endpoint (`main.py`):*
```python
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
```

### 1.2. API Key Generation

Once logged in, users can generate API keys. These keys are used to authenticate subsequent requests for uploading documents and making queries.

*Relevant Endpoint (`main.py`):*
```python
@app.post("/generate-key")
async def generate_new_api_key(user_id: str = Depends(get_current_user), name: Optional[str] = None):
    """Generate a new API key for the authenticated user."""
    api_key = generate_api_key(user_id, name)
    return APIResponse(success=True, message="API Key generated successfully", data={"api_key": api_key, "name": name})
```

## 2. Multi-Tenant Architecture

The core of the multi-tenant design is the per-user index partitioning in the FAISS vector store. Each user's data is stored in a dedicated subdirectory, ensuring complete data isolation.

### 2.1. Directory Structure

The vector store maintains a separate directory for each user, identified by their unique `user_id`.

- **Base Directory**: `data/faiss_index/`
- **User-Specific Directory**: `data/faiss_index/{user_id}/`

Each user-specific directory contains its own `faiss.index` and `documents.pkl` files.

### 2.2. User-Specific Index Management

The `FAISSVectorStore` class manages the user-specific indexes. It dynamically loads or creates an index for a user based on their `user_id`.

*Relevant Code (`vector_store.py`):*
```python
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
        # ... logic to load or create index ...
```

## 3. Data Ingestion Flow

Users can upload documents as plain text or files. The system processes these documents and stores them in the user's dedicated vector store.

### 3.1. Document Upload

The upload endpoints require a valid API key. The `user_id` is extracted from the API key to identify the correct index partition.

*Relevant Endpoint (`main.py`):*
```python
@app.post("/upload/file")
async def upload_file_document(user_id: str = Depends(get_current_user_id_from_api_key), file: UploadFile = File(...)):
    """Upload and process various file formats for the current user."""
    try:
        # ... file processing logic ...
        
        # Add to vector store for the specific user
        success = vector_store.add_documents(user_id, processed_data)
        
        # ... response logic ...
```

### 3.2. Storing Documents

The `FAISSVectorStore`'s `add_documents` method ensures that the processed data is added to the correct user's index.

*Relevant Code (`vector_store.py`):*
```python
def add_documents(self, user_id: str, processed_data: Dict[str, Any]) -> bool:
    """Add processed document data to a user's vector store."""
    logger.info(f"Adding {processed_data['total_chunks']} new chunks to the vector store for user {user_id}.")
    try:
        index, documents = self._load_index(user_id)
        
        # ... logic to add embeddings and documents ...
        
        self._save_index(user_id)
        
        logger.info(f"Documents added successfully for user {user_id}.")
        return True
    # ... error handling ...
```

## 4. Query Flow

When a user submits a query, the system retrieves relevant documents from the user's index, generates an answer using a language model, and returns the response along with the sources.

### 4.1. Submitting a Query

The query endpoint also requires a valid API key. The `user_id` is extracted to ensure the search is performed on the correct index.

*Relevant Endpoint (`main.py`):*
```python
@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, user_id: str = Depends(get_current_user_id_from_api_key)):
    """
    Query the knowledge base for the current user and get AI-generated response.
    """
    try:
        # Process query through RAG pipeline for the specific user
        result = rag_orchestrator.process_query(request.query, user_id)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"][:request.max_results],
            confidence=1.0 if result["context_used"] else 0.0
        )
    # ... error handling ...
```

### 4.2. Processing the Query

The `RAGOrchestrator` processes the query by first fetching relevant documents from the user's vector store and then generating an answer.

*Relevant Code (`query_service.py`):*
```python
class RAGOrchestrator:
    # ... initialization ...
    
    def process_query(self, query: str, user_id: str) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline for a specific user.
        """
        logger.info(f"Processing query: {query} for user: {user_id}")
        try:
            initial_state = {
                "query": query,
                "user_id": user_id,
                "retrieved_docs": [],
                "context": "",
                "answer": ""
            }
            
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # ... prepare response ...
            return response
        # ... error handling ...
```

This multi-tenant architecture ensures that each user's data is securely isolated, providing a robust and scalable solution for the RAG API.
