# query_enhancer.py
import openai
from typing import List, Dict, Any

class QueryEnhancer:
    def __init__(self, api_key: str = None):
        """Initialize the query enhancer with OpenAI API."""
        if api_key:
            openai.api_key = api_key
    
    def enhance_query(self, original_query: str) -> Dict[str, Any]:
        """Expand the original query to improve retrieval quality."""
        prompt = f"""You are an AI assistant that helps improve search queries for a retrieval system.
Given a user query, generate:
1. An improved version of the query that is more specific and includes relevant keywords
2. 2-3 alternative phrasings of the query that might help with retrieval
3. A list of 3-5 most important keywords or entities from the query

Original Query: {original_query}

Output the results in the following format:
Improved Query: [improved version]
Alternative Phrasings:
- [alternative 1]
- [alternative 2]
Keywords: [keyword1], [keyword2], [keyword3]"""

        try:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=200,
                temperature=0.3
            )
            
            result_text = response.choices[0].text.strip()
            
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
        
        except Exception as e:
            return {
                "original_query": original_query,
                "improved_query": original_query,  # Fall back to original query
                "alternative_phrasings": [],
                "keywords": [],
                "success": False,
                "error": str(e)
            }