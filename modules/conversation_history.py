# conversation_history.py
from typing import List, Dict, Any
from datetime import datetime

class ConversationHistory:
    def __init__(self, max_history: int = 10):
        """Initialize the conversation history manager.
        
        Args:
            max_history: Maximum number of conversation turns to keep in history.
        """
        self.max_history = max_history
        self.history = []
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    
    def add_interaction(self, user_message: str, system_response: str, 
                        retrieved_docs: List[Dict[str, Any]] = None) -> None:
        """Add a new interaction to the conversation history.
        
        Args:
            user_message: The message from the user
            system_response: The response from the system
            retrieved_docs: Optional list of retrieved documents used for the response
        """
        timestamp = datetime.now().isoformat()
        
        # Create a summary of retrieved documents for context
        doc_summary = []
        if retrieved_docs:
            for doc in retrieved_docs:
                source = doc.get("metadata", {}).get("source", "Unknown")
                similarity = doc.get("similarity", 0)
                doc_summary.append({
                    "source": source,
                    "similarity": similarity
                })
        
        # Create the interaction record
        interaction = {
            "timestamp": timestamp,
            "user_message": user_message,
            "system_response": system_response,
            "retrieved_docs": doc_summary
        }
        
        # Add to history
        self.history.append(interaction)
        
        # Trim history if it exceeds the maximum
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_chat_history(self, format_type: str = "text") -> str:
        """Get the conversation history in the specified format.
        
        Args:
            format_type: Format of the history ('text' or 'json')
        
        Returns:
            String representation of the conversation history
        """
        if format_type == "json":
            import json
            return json.dumps(self.history, indent=2)
        
        # Default to text format
        formatted_history = ""
        for i, interaction in enumerate(self.history):
            formatted_history += f"User: {interaction['user_message']}\n"
            formatted_history += f"Assistant: {interaction['system_response']}\n\n"
        
        return formatted_history.strip()
    
    def get_messages_for_llm(self, include_last_n: int = None) -> List[Dict[str, str]]:
        """Get the conversation history formatted for LLM context.
        
        Args:
            include_last_n: Only include the last N interactions (None = all)
        
        Returns:
            List of message dictionaries in the format expected by LLMs
        """
        messages = []
        
        # Determine how many history items to include
        history_to_use = self.history
        if include_last_n is not None and include_last_n < len(self.history):
            history_to_use = self.history[-include_last_n:]
        
        # Format as messages
        for interaction in history_to_use:
            messages.append({"role": "user", "content": interaction["user_message"]})
            messages.append({"role": "assistant", "content": interaction["system_response"]})
        
        return messages
    
    def clear(self) -> None:
        """Clear the conversation history."""
        self.history = []
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    
    def save_to_file(self, file_path: str = None) -> str:
        """Save the conversation history to a file.
        
        Args:
            file_path: Path to save the file (default: auto-generated based on session_id)
        
        Returns:
            Path to the saved file
        """
        import json
        import os
        
        if file_path is None:
            # Create a directory for conversation logs if it doesn't exist
            os.makedirs("conversation_logs", exist_ok=True)
            file_path = f"conversation_logs/conversation_{self.session_id}.json"
        
        with open(file_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return file_path