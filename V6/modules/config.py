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
MAX_TOKENS = 700

# Chat settings
MAX_HISTORY_TURNS = 10  # Number of conversation turns to remember
# LLM_MODEL = "gpt-3.5-turbo"
LLM_MODEL = "gpt-4o"
  # For chat API (ChatCompletion)
# LLM_MODEL = "gpt-3.5-turbo-instruct"  # For completion API

# API key (from environment variable)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")