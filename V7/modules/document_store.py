# document_store.py
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

class DocumentStore:
    def __init__(self, persist_directory: str = "chroma_db"):
        """Initialize the document store with ChromaDB."""
        self.persist_directory = persist_directory
        
        # Create the directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Get or create a collection for storing documents
        self.collection = self.client.get_or_create_collection(name="documents")
    
    def add_documents(self, document_chunks: List[Dict[str, Any]]):
        """Add document chunks to the vector store."""
        # Check if collection already has documents
        collection_count = self.collection.count()
        print(f"Current document count in collection: {collection_count}")
        
        # Generate unique IDs
        ids = [f"{chunk['metadata']['source']}_{chunk['metadata']['chunk_id']}" for chunk in document_chunks]
        documents = [chunk["text"] for chunk in document_chunks]
        embeddings = [chunk["embedding"] for chunk in document_chunks]
        metadatas = [chunk["metadata"] for chunk in document_chunks]
        
        # Add documents to the collection
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            print(f"Successfully added {len(documents)} documents to the collection.")
        except Exception as e:
            print(f"Error adding documents to collection: {e}")
            # Try adding them one by one
            success_count = 0
            for i in range(len(ids)):
                try:
                    self.collection.add(
                        ids=[ids[i]],
                        documents=[documents[i]],
                        embeddings=[embeddings[i]],
                        metadatas=[metadatas[i]]
                    )
                    success_count += 1
                except Exception as e2:
                    print(f"Error adding document {ids[i]}: {e2}")
            print(f"Added {success_count} out of {len(ids)} documents individually.")
    
    def similarity_search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the most similar documents for a query embedding."""
        # Check collection size
        collection_count = self.collection.count()
        if collection_count == 0:
            print("Warning: Document collection is empty. No results will be returned.")
            return []
        
        print(f"Searching among {collection_count} documents with top_k={top_k}")
        
        # Perform the query
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection_count)
        )
        
        retrieved_docs = []
        
        # Debug the results
        print(f"Query returned {len(results['ids'][0])} documents.")
        print(f"Result keys: {list(results.keys())}")
        
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                doc = {
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if 'metadatas' in results and results['metadatas'][0] else {},
                }
                
                # Add distance/score if available
                if 'distances' in results and len(results['distances']) > 0:
                    doc["score"] = results['distances'][0][i]
                
                retrieved_docs.append(doc)
        
        return retrieved_docs






