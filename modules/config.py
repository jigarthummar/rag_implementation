# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Application settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 80
EMBEDDING_MODEL = 'intfloat/e5-base-v2' # "all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "chroma_db"
TOP_K = 7
SIMILARITY_THRESHOLD = 0.7
MAX_TOKENS = 4000

# Chat settings
MAX_HISTORY_TURNS = 10  

# LLM models
LLM_MODEL = "openai/gpt-4o-2024-11-20"

# OpenRouter models
QUERY_ENHANCER_MODEL = "anthropic/claude-3-haiku"  # Lighter model for query enhancement

# API keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Keep for backward compatibility if needed
