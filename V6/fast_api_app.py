# fast_api_app.py
import os
import sys
import gc
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import shutil
import logging
import asyncio
from contextlib import asynccontextmanager

# Import necessary modules from your RAG application
import modules.config as config
from modules.document_processor import DocumentProcessor
from modules.embeddings import EmbeddingGenerator
from modules.document_store import DocumentStore
from modules.query_enhancer import QueryEnhancer
from modules.retriever import Retriever
from modules.llm_interface import LLMInterface
from modules.conversation_history import ConversationHistory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Define response models
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    use_history: bool = True

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    conversation_id: str
    success: bool

class DocumentUploadResponse(BaseModel):
    filename: str
    success: bool
    message: str

# Initialize application components
def get_rag_app():
    return app.state.rag_app

class EnhancedRAGApplication:
    def __init__(self):
        """Initialize the enhanced RAG application with all its components."""
        # Initialize components
        logger.info("Initializing RAG application components...")
        
        self.document_processor = DocumentProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        self.embedding_generator = EmbeddingGenerator(
            model_name=config.EMBEDDING_MODEL
        )
        
        self.document_store = DocumentStore(
            persist_directory=config.PERSIST_DIRECTORY
        )
        
        try:
            # Initialize the query enhancer
            self.query_enhancer = QueryEnhancer(api_key=config.OPENAI_API_KEY)
            
            # Initialize the enhanced retriever
            self.retriever = Retriever(
                embedding_generator=self.embedding_generator,
                document_store=self.document_store,
                query_enhancer=self.query_enhancer,
                top_k=config.TOP_K,
                similarity_threshold=config.SIMILARITY_THRESHOLD,
                use_query_enhancement=True,
                max_docs_per_source=7
            )
            
            # Initialize the enhanced LLM interface with chat capabilities
            self.llm = LLMInterface(
                api_key=config.OPENAI_API_KEY,
                model=config.LLM_MODEL
            )
            
            # Dictionary to store conversation histories by ID
            self.conversations = {}
            
            logger.info("RAG application components initialized successfully")
            
        except ValueError as e:
            logger.error(f"Error initializing components: {e}")
            logger.error("Please set your OPENAI_API_KEY in a .env file or as an environment variable.")
            sys.exit(1)
    
    def get_or_create_conversation(self, conversation_id: Optional[str] = None) -> tuple:
        """Get an existing conversation or create a new one."""
        if conversation_id and conversation_id in self.conversations:
            return conversation_id, self.conversations[conversation_id]
        
        # Create a new conversation
        from datetime import datetime
        new_id = conversation_id or datetime.now().strftime("%Y%m%d%H%M%S")
        self.conversations[new_id] = ConversationHistory(
            max_history=config.MAX_HISTORY_TURNS
        )
        return new_id, self.conversations[new_id]
    
    async def upload_document(self, file_path: str) -> bool:
        """Process and upload a document to the vector store."""
        try:
            # Check if the file exists
            if not os.path.exists(file_path):
                logger.error(f"Error: File not found: {file_path}")
                return False
                
            # Check the file size to determine processing method
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            
            # Process the document
            logger.info(f"Processing document: {file_path}, Size= {file_size:.2f}MB")
            
            # This could be CPU intensive, use thread pool
            doc_chunks = await asyncio.to_thread(
                self.document_processor.process_document,
                file_path
            )
                
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(doc_chunks)} chunks...")
            doc_chunks_with_embeddings = await asyncio.to_thread(
                self.embedding_generator.process_document_chunks,
                doc_chunks
            )
                
            # Add to vector store
            logger.info("Adding to vector store...")
            await asyncio.to_thread(
                self.document_store.add_documents,
                doc_chunks_with_embeddings
            )
                
            logger.info(f"Successfully uploaded document: {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error uploading document: {e}")
            return False
        finally:
            # Force garbage collection
            gc.collect()
    
    async def process_message(self, message: str, conversation_id: Optional[str] = None, use_history: bool = True) -> Dict[str, Any]:
        """
        Process a user message in the context of a conversation.
        
        Args:
            message: User's message
            conversation_id: ID of the conversation
            use_history: Whether to incorporate conversation history
        
        Returns:
            Dictionary with the system response and related information
        """
        try:
            # Get or create conversation history
            conv_id, conversation = self.get_or_create_conversation(conversation_id)
            
            # Retrieve relevant documents with the retriever
            logger.info("Retrieving relevant information...")
            retrieved_docs = await asyncio.to_thread(
                self.retriever.retrieve,
                message
            )
            
            if not retrieved_docs:
                response_text = "I couldn't find any relevant information to answer your question."
                
                # Add to conversation history
                conversation.add_interaction(
                    user_message=message,
                    system_response=response_text,
                    retrieved_docs=[]
                )
                
                return {
                    "response": response_text,
                    "sources": [],
                    "conversation_id": conv_id,
                    "success": True
                }
            
            # Get conversation history for context if needed
            chat_history = None
            if use_history and len(conversation.history) > 0:
                chat_history = conversation.get_messages_for_llm()
            
            # Generate response using the enhanced LLM interface
            logger.info("Generating response...")
            result = await asyncio.to_thread(
                self.llm.generate_response,
                query=message,
                context_docs=retrieved_docs,
                max_tokens=config.MAX_TOKENS,
                conversation_history=chat_history
            )
            
            # Add source information to the result
            sources = []
            for doc in result.get("used_context", []):
                if "metadata" in doc:
                    source = doc["metadata"].get("source", "unknown")
                    page_range = doc["metadata"].get("page_range", "")
                    if page_range:
                        source = f"{source} (Pages {page_range})"
                    sources.append(source)
            
            # Add to conversation history
            conversation.add_interaction(
                user_message=message,
                system_response=result.get("response", ""),
                retrieved_docs=retrieved_docs
            )
            
            return {
                "response": result.get("response", ""),
                "sources": list(set(sources)),  # Remove duplicates
                "conversation_id": conv_id,
                "success": result.get("success", False)
            }
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_response = f"Error processing your message: {e}"
            
            # Add error to conversation history if we have a valid conversation
            if conversation_id in self.conversations:
                self.conversations[conversation_id].add_interaction(
                    user_message=message,
                    system_response=error_response,
                    retrieved_docs=[]
                )
            
            return {
                "response": error_response,
                "sources": [],
                "conversation_id": conversation_id or "error",
                "success": False
            }
        finally:
            # Force garbage collection
            gc.collect()

# FastAPI application startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize RAG application
    logger.info("Starting RAG application...")
    app.state.rag_app = EnhancedRAGApplication()
    yield
    # Shutdown: Clean up resources
    logger.info("Shutting down RAG application...")
    # Any cleanup code goes here

# Create the FastAPI application
app = FastAPI(
    title="RAG Chat API",
    description="API for document ingestion and RAG-based chat",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint for uploading documents
@app.post("/upload/", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the RAG system."""
    if not file.filename.lower().endswith(('.pdf', '.txt')):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF and TXT files are supported"
        )
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    try:
        # Write the uploaded file content to the temporary file
        with temp_file as f:
            shutil.copyfileobj(file.file, f)
        
        # Process and upload the document
        rag_app = get_rag_app()
        success = await rag_app.upload_document(temp_file.name)
        
        if success:
            return DocumentUploadResponse(
                filename=file.filename,
                success=True,
                message=f"Document '{file.filename}' successfully uploaded and processed"
            )
        else:
            return DocumentUploadResponse(
                filename=file.filename,
                success=False,
                message=f"Failed to process document '{file.filename}'"
            )
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

# Endpoint for chat
@app.post("/chat/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to the RAG chat system."""
    try:
        rag_app = get_rag_app()
        result = await rag_app.process_message(
            message=request.message,
            conversation_id=request.conversation_id,
            use_history=request.use_history
        )
        
        return ChatResponse(
            response=result["response"],
            sources=result["sources"],
            conversation_id=result["conversation_id"],
            success=result["success"]
        )
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat message: {str(e)}"
        )

# Health check endpoint
@app.get("/health/")
async def health_check():
    """Check if the API is running."""
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fast_api_app:app", host="0.0.0.0", port=8000, reload=True)