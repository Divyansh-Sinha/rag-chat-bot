import openai
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
from config import config
from embedding_service import embedding_service
from vector_store import vector_store
from logging_config import logger

class DataFetchingTool:
    """Tool for fetching additional data from configured sources"""
    
    def __init__(self):
        self.data_sources = config.get_data_source_config()
    
    def fetch_data(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch additional data based on configuration for a specific user.
        """
        user_id = state['user_id']
        logger.info(f"Fetching data for query: {state['query']} for user: {user_id}")
        try:
            # Create embedding for the query (returns list, take first element)
            query_embeddings = embedding_service.create_embeddings([state["query"]])
            query_embedding = query_embeddings[0]  # Extract single embedding
            
            # Search in vector store for the specific user
            results = vector_store.search(user_id, query_embedding, k=5)
            logger.info(f"Found {len(results)} relevant documents for user {user_id}.")
            
            # Update state with results
            state["retrieved_docs"] = results
            
            # Create context from retrieved documents
            context_parts = []
            for doc in results:
                context_parts.append(f"Content: {doc['chunk']}")
            
            state["context"] = "\n\n".join(context_parts)
            
            return state
        except Exception as e:
            logger.info(f"Data fetching error for user {user_id}: {str(e)}", exc_info=True)
            state["retrieved_docs"] = []
            state["context"] = ""
            return state


class QueryAnsweringTool:
    """Tool for generating answers using retrieved context"""
    
    def __init__(self):
        openai.api_key = config.OPENAI_API_KEY
    
    def generate_answer(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate answer using OpenAI GPT model with retrieved context
        """
        logger.info("Generating answer...")
        try:
            # Prepare prompt with context and query
            prompt = f"""
            Based on the following context, answer the user's question. If the context doesn't contain relevant information, say so.

            Context:
            {state["context"]}

            Question: {state["query"]}

            Answer:
            """
            
            # Call OpenAI API
            response = openai.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            state["answer"] = response.choices[0].message.content
            logger.info("Answer generated successfully.")
            return state
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}", exc_info=True)
            state["answer"] = f"Error generating answer: {str(e)}"
            return state


class RAGOrchestrator:
    """LangGraph orchestrator for RAG workflow"""
    
    def __init__(self):
        self.data_fetcher = DataFetchingTool()
        self.answer_generator = QueryAnsweringTool()
        self.workflow = self._create_workflow()
    
    def _create_workflow(self):
        """Create LangGraph workflow"""
        # Create state graph with dict as state type
        workflow = StateGraph(dict)
        
        # Add nodes for our tools
        workflow.add_node("fetch_data", self.data_fetcher.fetch_data)
        workflow.add_node("generate_answer", self.answer_generator.generate_answer)
        
        # Define the workflow edges
        workflow.set_entry_point("fetch_data")
        workflow.add_edge("fetch_data", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        return workflow.compile()
    
    def process_query(self, query: str, user_id: str) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline for a specific user.
        """
        logger.info(f"Processing query: {query} for user: {user_id}")
        try:
            # Initialize state as a dict
            initial_state = {
                "query": query,
                "user_id": user_id,
                "retrieved_docs": [],
                "context": "",
                "answer": ""
            }
            
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Prepare response
            response = {
                "answer": final_state.get("answer", ""),
                "sources": [
                    {
                        "content": doc["chunk"][:200] + "...",  # Truncate for display
                        "metadata": doc["metadata"],
                        "similarity_score": doc.get("similarity_score", 0)
                    }
                    for doc in final_state.get("retrieved_docs", [])
                ],
                "context_used": len(final_state.get("retrieved_docs", [])) > 0
            }
            
            logger.info(f"Query processed successfully for user {user_id}.")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query for user {user_id}: {str(e)}", exc_info=True)
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "context_used": False
            }


# Initialize RAG orchestrator
rag_orchestrator = RAGOrchestrator()