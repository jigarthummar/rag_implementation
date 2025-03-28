# query_enhancer.py
import requests
import json
import os
from typing import List, Dict, Any

class QueryEnhancer:
    def __init__(self, api_key: str = None, model: str = "anthropic/claude-3-haiku"):
        """Initialize the query enhancer with OpenRouter API."""
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key is required")
        
        # OpenRouter API endpoint
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Use a smaller, faster model for query enhancement
        self.model = model
    
    def enhance_query(self, original_query: str) -> Dict[str, Any]:
        """Expand the original query to improve retrieval quality."""
        system_prompt = """You are an AI assistant that helps improve search queries for a retrieval system."""
        
        user_prompt = f"""Given a user query, generate:
1. An improved version of the query that is more specific and includes relevant keywords
2. 2-3 alternative phrasings of the query that might help with retrieval
3. A list of 3-5 most important keywords or entities from the query
4. Break down the query into subqueries that are more specific and easier to retrieve.

Original Query: {original_query}

Output the results in the following format:
Improved Query: [improved version]
Alternative Phrasings:
- [alternative 1]
- [alternative 2]
- [alternative 3]
Keywords: [keyword1], [keyword2], [keyword3]"""

        try:
            # Prepare the request payload
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.3
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
                result_text = response_data["choices"][0]["message"]["content"].strip()
                
                # Parse the response
                improved_query = ""
                alt_phrasings = []
                keywords = []
                
                for line in result_text.split('\n'):
                    if line.startswith("Improved Query:"):
                        improved_query = line.replace("Improved Query:", "").strip()
                    elif line.startswith("- ") and "Alternative Phrasings" in result_text:
                        alt_phrasings.append(line.replace("- ", "").strip())
                    elif line.startswith("Keywords:"):
                        keywords_text = line.replace("Keywords:", "").strip()
                        keywords = [k.strip() for k in keywords_text.split(',')]
                
                return {
                    "original_query": original_query,
                    "improved_query": improved_query,
                    "alternative_phrasings": alt_phrasings,
                    "keywords": keywords,
                    "success": True
                }
            else:
                return {
                    "original_query": original_query,
                    "improved_query": original_query,  # Fall back to original query
                    "alternative_phrasings": [],
                    "keywords": [],
                    "success": False,
                    "error": f"API Error: {response.status_code} - {response.text}"
                }
        
        except Exception as e:
            return {
                "original_query": original_query,
                "improved_query": original_query,  # Fall back to original query
                "alternative_phrasings": [],
                "keywords": [],
                "success": False,
                "error": str(e)
            }