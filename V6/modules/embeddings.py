from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import torch

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = None):
        """
        Initialize the embedding generator with a SentenceTransformer model.

        Args:
            model_name: The name of the SentenceTransformer model to use
            device: The device to use (cuda, mps, cpu). If None, will use best available
        """
        # Automatically use best available device unless explicitly specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
                device = "mps"  # For Apple Silicon (M1/M2/M3)
            else:
                device = "cpu"

        self.device = device
        self.model = SentenceTransformer(model_name)
        self.model.to(device)

        # Print device information for confirmation
        if device == "cuda" and torch.cuda.is_available():
            print(f"Using GPU (CUDA): {torch.cuda.get_device_name(0)}")
        elif device == "mps":
            print("Using GPU (Apple Silicon)")
        else:
            print("Using CPU")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings for a single text."""
        return self.model.encode(text, normalize_embeddings=True, device=self.device).tolist()

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        For large batches on GPU, consider using show_progress_bar=True
        """
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            device=self.device,
            batch_size=32,  # Adjust batch size based on your memory
            show_progress_bar=True
        ).tolist()

    def process_document_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process document chunks by adding embeddings."""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.generate_embeddings(texts)

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i]

        return chunks