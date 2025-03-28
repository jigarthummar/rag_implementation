# document_processor.py
import os
from typing import List, Dict, Union
import fitz  # PyMuPDF
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file using PyMuPDF for better results."""
        # Open the PDF
        doc = fitz.open(file_path)
        total_pages = len(doc)
        
        # Extract text from all pages
        full_text = ""
        for i in range(total_pages):
            page = doc[i]
            page_text = page.get_text()
            
            # Basic text cleaning
            page_text = re.sub(r'\s+', ' ', page_text).strip()
            
            full_text += page_text + " "
        
        # Close the document
        doc.close()
        
        return full_text
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from various file types."""
        if file_path.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess the extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Additional preprocessing steps can be added here
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into smaller chunks using LangChain's TokenTextSplitter
        for token-based chunking instead of character-based.
        """
        from langchain.text_splitter import TokenTextSplitter
        
        text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            encoding_name="cl100k_base"  # Default tokenizer for GPT-4, GPT-3.5-Turbo
        )
        chunks = text_splitter.split_text(text)
        return chunks


    # def chunk_text(self, text: str) -> List[str]:
    #     """
    #     Split text into smaller chunks using LangChain's RecursiveCharacterTextSplitter
    #     for more intelligent chunking based on natural text boundaries.
    #     """
    #     text_splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=self.chunk_size,
    #         chunk_overlap=self.chunk_overlap,
    #         length_function=len,
    #         separators=["\n\n", "\n", ". ", "", "!", "?"]
    #     )
    #     chunks = text_splitter.split_text(text)
    #     return chunks
    
    def process_document(self, file_path: str, show_progress: bool = False) -> List[Dict[str, str]]:
        """Process a document: extract, preprocess, and chunk its text."""
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Extract text from the document
        text = self.extract_text(file_path)
        
        # Preprocess the text
        text = self.preprocess_text(text)
        
        # Split the text into chunks
        chunks = self.chunk_text(text)
        
        # Create document chunks with metadata
        doc_chunks = []
        for i, chunk in enumerate(chunks):
            doc_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": file_path,
                    "chunk_id": i
                }
            })
        
        return doc_chunks