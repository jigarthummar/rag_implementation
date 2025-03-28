# app.py
import os
import sys
import gc
from typing import List, Dict, Any
import modules.config as config
from modules.document_processor import DocumentProcessor
from modules.embeddings import EmbeddingGenerator
from modules.document_store import DocumentStore
from modules.query_enhancer import QueryEnhancer
from modules.retriever import Retriever
from modules.llm_interface import LLMInterface
from modules.conversation_history import ConversationHistory

class EnhancedRAGChatApplication:
    def __init__(self):
        """Initialize the enhanced RAG chat application with all its components."""
        # Initialize components
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
                model=config.LLM_MODEL  # Use the model configuration from config
            )
            
            # Initialize conversation history manager
            self.conversation_history = ConversationHistory(
                max_history=config.MAX_HISTORY_TURNS  # Use the config value
            )
        except ValueError as e:
            print(f"Error initializing components: {e}")
            print("Please set your OPENAI_API_KEY in a .env file or as an environment variable.")
            sys.exit(1)
    
    def upload_document(self, file_path: str) -> bool:
        """Process and upload a document to the vector store."""
        try:
            # Check if the file exists
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                return False
                
            # Check the file size to determine processing method
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            
            # For smaller files, use the regular document processor
            print(f"Processing document: {file_path}, Size= {file_size:.2f}MB")
            doc_chunks = self.document_processor.process_document(file_path)
                
            # Generate embeddings
            print(f"Generating embeddings for {len(doc_chunks)} chunks...")
            doc_chunks_with_embeddings = self.embedding_generator.process_document_chunks(doc_chunks)
                
            # Add to vector store
            print("Adding to vector store...")
            self.document_store.add_documents(doc_chunks_with_embeddings)
                
            print(f"Successfully uploaded document: {file_path}")
            return True
        
        except Exception as e:
            print(f"Error uploading document: {e}")
            return False
        finally:
            # Force garbage collection
            gc.collect()
    
    def process_message(self, message: str, use_history: bool = True) -> Dict[str, Any]:
        """
        Process a user message in the context of the current conversation.
        
        Args:
            message: User's message
            use_history: Whether to incorporate conversation history
        
        Returns:
            Dictionary with the system response and related information
        """
        try:
            # Retrieve relevant documents with the retriever
            print("Retrieving relevant information...")
            retrieved_docs = self.retriever.retrieve(message)
            
            if not retrieved_docs:
                response_text = "I couldn't find any relevant information to answer your question."
                
                # Add to conversation history
                self.conversation_history.add_interaction(
                    user_message=message,
                    system_response=response_text,
                    retrieved_docs=[]
                )
                
                return {
                    "response": response_text,
                    "sources": [],
                    "success": True
                }
            
            # Get conversation history for context if needed
            chat_history = None
            if use_history and len(self.conversation_history.history) > 0:
                chat_history = self.conversation_history.get_messages_for_llm()
            
            # Generate response using the enhanced LLM interface
            print("Generating response...")
            result = self.llm.generate_response(
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
            self.conversation_history.add_interaction(
                user_message=message,
                system_response=result.get("response", ""),
                retrieved_docs=retrieved_docs
            )
            
            return {
                "response": result.get("response", ""),
                "sources": list(set(sources)),  # Remove duplicates
                "success": result.get("success", False)
            }
        except Exception as e:
            error_response = f"Error processing your message: {e}"
            
            # Add error to conversation history
            self.conversation_history.add_interaction(
                user_message=message,
                system_response=error_response,
                retrieved_docs=[]
            )
            
            return {
                "response": error_response,
                "sources": [],
                "success": False
            }
        finally:
            # Force garbage collection
            gc.collect()
    
    def run_chat_cli(self):
        """Run the chat-based command-line interface for the RAG application."""
        print("=" * 50)
        print("Welcome to the RAG Chat Application!")
        print("=" * 50)
        print("\nCommands:")
        print("- To upload documents: 'upload <file_path>'")
        print("- To start a new conversation: 'new'")
        print("- To save conversation: 'save [file_path]'")
        print("- To exit: 'exit' or 'quit'")
        print("\nStart chatting by typing your message!")
        
        in_chat_session = False
        
        while True:
            try:
                if not in_chat_session:
                    user_input = input("\n> ").strip()
                else:
                    user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ["exit", "quit"]:
                    if len(self.conversation_history.history) > 0:
                        save_response = input("Do you want to save this conversation before exiting? (y/n): ").strip().lower()
                        if save_response == 'y':
                            file_path = self.conversation_history.save_to_file()
                            print(f"Conversation saved to: {file_path}")
                    
                    print("Thank you for using the RAG Chat Application. Goodbye!")
                    break
                
                elif user_input.lower().startswith("upload "):
                    file_path = user_input[7:].strip()
                    self.upload_document(file_path)
                    # Force memory cleanup after upload
                    gc.collect()
                
                elif user_input.lower() == "new":
                    if len(self.conversation_history.history) > 0:
                        save_response = input("Do you want to save the current conversation first? (y/n): ").strip().lower()
                        if save_response == 'y':
                            file_path = self.conversation_history.save_to_file()
                            print(f"Conversation saved to: {file_path}")
                    
                    self.conversation_history.clear()
                    print("Starting a new conversation!")
                    in_chat_session = True
                
                elif user_input.lower().startswith("save"):
                    parts = user_input.split(" ", 1)
                    file_path = parts[1].strip() if len(parts) > 1 else None
                    
                    saved_path = self.conversation_history.save_to_file(file_path)
                    print(f"Conversation saved to: {saved_path}")
                
                else:
                    # Process the user message
                    result = self.process_message(user_input)
                    
                    if result["success"]:
                        print("\nAssistant:", result["response"])
                        
                        if result["sources"]:
                            print("\nSources:")
                            for source in result["sources"]:
                                print(f"- {source}")
                    else:
                        print(f"\nError: {result['response']}")
                    
                    # We're now in a chat session
                    in_chat_session = True
                    
                    # Force memory cleanup after processing
                    gc.collect()
            
            except KeyboardInterrupt:
                print("\nOperation interrupted by user.")
                if input("\nDo you want to exit? (y/n): ").strip().lower() == 'y':
                    break
            except Exception as e:
                print(f"\nAn unexpected error occurred: {e}")
                # Force memory cleanup after error
                gc.collect()

if __name__ == "__main__":
    app = EnhancedRAGChatApplication()
    app.run_chat_cli()