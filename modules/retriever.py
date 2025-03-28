# retriever.py
from typing import List, Dict, Any
from modules.embeddings import EmbeddingGenerator
from modules.document_store import DocumentStore
from modules.query_enhancer import QueryEnhancer  # Updated import
import numpy as np

class Retriever:
    def __init__(self, 
                embedding_generator: EmbeddingGenerator, 
                document_store: DocumentStore,
                query_enhancer: QueryEnhancer = None,
                top_k: int = 5,
                similarity_threshold: float = 0.7,
                use_query_enhancement: bool = True,
                max_docs_per_source: int = 2):
        """Initialize the enhanced retriever with embedding generator and document store."""
        self.embedding_generator = embedding_generator
        self.document_store = document_store
        self.query_enhancer = query_enhancer
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.use_query_enhancement = use_query_enhancement
        self.max_docs_per_source = max_docs_per_source
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using enhanced query techniques.
        Uses query expansion and multi-query retrieval for better results.
        """
        all_docs = []
        used_queries = [query]
        
        # Use query enhancement if available
        if self.query_enhancer and self.use_query_enhancement:
            print("Enhancing query...")
            enhanced = self.query_enhancer.enhance_query(query)
            
            if enhanced["success"]:
                # Add the improved query
                if enhanced["improved_query"] and enhanced["improved_query"] != query:
                    used_queries.append(enhanced["improved_query"])
                
                # Add alternative phrasings
                for alt_query in enhanced["alternative_phrasings"]:
                    if alt_query not in used_queries:
                        used_queries.append(alt_query)
                
                print(f"Enhanced original query: '{query}' to:")
                for q in used_queries:
                    print(f"- '{q}'")
        
        # Retrieve documents for each query version
        for query_version in used_queries:
            query_embedding = self.embedding_generator.generate_embedding(query_version)
            retrieved_docs = self.document_store.similarity_search(
                query_embedding=query_embedding,
                top_k=self.top_k
            )
            
            if retrieved_docs:
                print(f"Retrieved {len(retrieved_docs)} documents for query: '{query_version}'")
                all_docs.extend(retrieved_docs)
        
        # Remove duplicates (based on text content)
        unique_docs = []
        seen_texts = set()
        
        for doc in all_docs:
            text = doc["text"]
            # Use a hash of the text to avoid long string comparisons
            text_hash = hash(text)
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_docs.append(doc)
        
        print(f"After removing duplicates: {len(unique_docs)} unique documents")
        
        # Calculate similarity scores for ranking
        query_embedding = self.embedding_generator.generate_embedding(query)  # Use original query for scoring
        for doc in unique_docs:
            if "embedding" in doc:
                # Calculate cosine similarity if embeddings are available
                similarity = np.dot(query_embedding, doc["embedding"]) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc["embedding"])
                )
                print(similarity)
                doc["similarity"] = float(similarity)
            elif "score" in doc:
                # If score is a distance, convert to similarity
                if doc["score"] >= 0 and doc["score"] <= 2:
                    doc["similarity"] = 1 - (doc["score"] / 2)
                else:
                    doc["similarity"] = doc["score"]
        
        # Sort by similarity score
        unique_docs.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        # Filter by similarity threshold
        filtered_docs = [doc for doc in unique_docs if doc.get("similarity", 0) >= self.similarity_threshold]
        
        # If we filtered out everything but had results, lower the threshold
        if unique_docs and not filtered_docs:
            print("Warning: All documents were filtered out. Using top documents as fallback.")
            filtered_docs = unique_docs[:min(5, len(unique_docs))]
        
        # Ensure diversity of sources
        if self.max_docs_per_source > 0:
            source_count = {}
            diverse_docs = []
            
            for doc in filtered_docs:
                source = doc.get("metadata", {}).get("source", "unknown")
                source_count[source] = source_count.get(source, 0) + 1
                
                if source_count[source] <= self.max_docs_per_source:
                    diverse_docs.append(doc)
            
            filtered_docs = diverse_docs
        
        print(f"Returning {len(filtered_docs)} filtered documents.")
        return filtered_docs