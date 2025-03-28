# llm_interface.py
import os
import openai
from typing import List, Dict, Any, Optional

class LLMInterface:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """Initialize the enhanced LLM interface.
        
        Args:
            api_key: OpenAI API key
            model: The model to use for chat completion
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Set the API key for the openai module
        openai.api_key = self.api_key
        
        # Set the default model
        self.model = model
        
        # Whether to use the chat completion API (for newer models) 
        # or the completion API (for older models)
        self.use_chat_api = not model.endswith("-instruct")
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]], 
                         max_tokens: int = 500, 
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
            if self.use_chat_api:
                # Prepare the messages for the chat API
                messages = [{"role": "system", "content": system_instructions}]
                
                # Add conversation history if available
                if conversation_history:
                    messages.extend(conversation_history)
                
                # Add the current query with context
                user_content = f"Context documents:\n{context}\n\nQuestion: {query}"
                messages.append({"role": "user", "content": user_content})
                
                # Make the API call
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,  # Lower temperature for more focused answers
                    top_p=0.9
                )
                
                return {
                    "response": response.choices[0].message['content'].strip(),
                    "used_context": context_docs,
                    "success": True
                }
            else:
                # For older models using the completions API
                # Build the prompt with conversation history if available
                history_text = ""
                if conversation_history:
                    for msg in conversation_history:
                        role = msg["role"]
                        content = msg["content"]
                        if role == "user":
                            history_text += f"User: {content}\n"
                        elif role == "assistant":
                            history_text += f"Assistant: {content}\n"
                
                if history_text:
                    history_text = "Previous conversation:\n" + history_text + "\n"
                
                prompt = f"""{system_instructions}

{history_text}Context documents:
{context}

Question: {query}

Answer:"""

                # Using the OpenAI completions API
                response = openai.Completion.create(
                    engine=self.model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=0.3,
                    top_p=0.9,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                
                return {
                    "response": response.choices[0].text.strip(),
                    "used_context": context_docs,
                    "success": True
                }
                
        except Exception as e:
            return {
                "response": f"Error generating response: {str(e)}",
                "used_context": [],
                "success": False
            }