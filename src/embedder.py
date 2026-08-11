"""
embedder.py — Embedding Engine for converting text queries into vector embeddings.

Uses Sentence Transformers to generate normalized dense vector embeddings for text inputs,
which are suitable for cosine similarity search (e.g. via inner product in FAISS).
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL
from src.logger import get_logger

log = get_logger(__name__)


class Embedder:
    """
    Wrapper around SentenceTransformer for generating normalized query embeddings.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the Embedder with a specified Sentence Transformer model.

        Args:
            model_name: Hugging Face model identifier (defaults to EMBEDDING_MODEL in config).
        """
        self.model_name = model_name
        log.info(f"Loading embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        log.info(f"Embedding model '{self.model_name}' loaded successfully.")

    def embed(self, text: str) -> np.ndarray:
        """
        Generate a 1D normalized float32 embedding vector for a single text string.

        Args:
            text: Input query string.

        Returns:
            1D numpy array of shape (dimension,) with dtype float32, L2-normalized.
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty or whitespace-only text.")

        # normalize_embeddings=True normalizes vectors to unit L2 length (||v|| = 1.0)
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate 2D normalized float32 embedding matrix for a list of text strings.

        Args:
            texts: List of input query strings.

        Returns:
            2D numpy array of shape (len(texts), dimension) with dtype float32, L2-normalized.
        """
        if not texts:
            raise ValueError("Input text list for batch embedding cannot be empty.")

        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Empty text at index {i} in batch.")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        """
        Return the dimension of the embedding vectors produced by this model.
        """
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()
