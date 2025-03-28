# llm_interface.py
import os
import requests
import json
from typing import List, Dict, Any, Optional

class LLMInterface:
    def __init__(self, api_key: str = None, model: str = "anthropic/claude-3-opus"):
        """Initialize the LLM interface for OpenRouter.
        
        Args:
            api_key: OpenRouter API key
            model: The model to use for completion (default: anthropic/claude-3-opus)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        # Set the default model
        self.model = model
        
        # OpenRouter API endpoint
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]], 
                         max_tokens: int = 4000, 
                         conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Generate a response to the query based on retrieved context documents.
        
        Args:
            query: The user's query
            context_docs: The retrieved documents
            max_tokens: Maximum tokens to generate
            conversation_history: Optional list of previous messages in the conversation
        
        Returns:
            Dictionary containing the response and related information
        """
        # Prepare the context with source information
        formatted_contexts = []
        
        for i, doc in enumerate(context_docs):
            source = doc.get("metadata", {}).get("source", "Unknown source")
            similarity = doc.get("similarity", "Unknown")
            
            # Format the context with source information - handle different similarity types
            if isinstance(similarity, float):
                formatted_context = f"[Document {i+1}] (Source: {source}, Relevance: {similarity:.2f})\n{doc['text']}"
            else:
                formatted_context = f"[Document {i+1}] (Source: {source}, Relevance: {similarity})\n{doc['text']}"
            formatted_contexts.append(formatted_context)
        
        # Join all contexts with clear separation
        context = "\n\n" + "\n\n".join(formatted_contexts)
        
        # Build system instructions
        system_instructions = """You are a helpful assistant tasked with answering questions based on specific provided documents.
Answer based ONLY on the provided context documents and conversation history.

INSTRUCTIONS:
1. Use ONLY the information from the provided context documents
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question"
3. Synthesize information from multiple documents when necessary
4. Don't make up or infer information that isn't explicitly stated in the documents
5. Structure your answer in a clear, concise way
6. Answer the questions in minimum 2 sentences.
7. Maintain continuity from previous conversation when relevant"""

        try:
            # Prepare the messages for the OpenRouter API
            messages = [{"role": "system", "content": system_instructions}]
            
            # Add conversation history if available
            if conversation_history:
                messages.extend(conversation_history)
            
            # Add the current query with context
            user_content = f"Context documents:\n{context}\n\nQuestion: {query}"
            messages.append({"role": "user", "content": user_content})
            
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.3,  # Lower temperature for more focused answers
                "top_p": 0.9
            }
            
            # Set up headers
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make the API call
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload)
            )
            
            # Handle the response
            if response.status_code == 200:
                response_data = response.json()
                return {
                    "response": response_data["choices"][0]["message"]["content"].strip(),
                    "used_context": context_docs,
                    "success": True
                }
            else:
                error_msg = f"OpenRouter API error: {response.status_code} - {response.text}"
                return {
                    "response": error_msg,
                    "used_context": [],
                    "success": False
                }
                
        except Exception as e:
            return {
                "response": f"Error generating response: {str(e)}",
                "used_context": [],
                "success": False
            }