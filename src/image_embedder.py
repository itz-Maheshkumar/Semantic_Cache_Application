"""
image_embedder.py — Embedding Engine for converting image queries into vector
embeddings, for the IMAGE side of multi-modal semantic caching.

Uses a CLIP model (via Sentence Transformers) to generate normalized dense
vector embeddings for images. CLIP's image and text towers share one vector
space, but this project only ever compares image-to-image (see
src/modality.py for why) — cross-modal text<->image lookup is not attempted.
"""

import io
from pathlib import Path
from typing import List, Union

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from src.config import IMAGE_EMBEDDING_MODEL
from src.logger import get_logger

log = get_logger(__name__)

# Anything embed()/embed_batch() will accept for a single image.
ImageInput = Union[str, Path, bytes, Image.Image]


class ImageEmbedder:
    """
    Wrapper around a CLIP SentenceTransformer model for generating normalized
    image embeddings. Mirrors src/embedder.py's Embedder interface
    (embed / embed_batch / embedding_dim) so SemanticCache can use either one
    interchangeably without caring which modality it's actually indexing.
    """

    def __init__(self, model_name: str = IMAGE_EMBEDDING_MODEL):
        """
        Initialize the ImageEmbedder with a specified CLIP model.

        Args:
            model_name: Hugging Face / Sentence-Transformers model identifier
                (defaults to IMAGE_EMBEDDING_MODEL in config, e.g. "clip-ViT-B-32").
        """
        self.model_name = model_name
        log.info(f"Loading image embedding model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        log.info(f"Image embedding model '{self.model_name}' loaded successfully.")

    @staticmethod
    def _load_image(image: ImageInput) -> Image.Image:
        """
        Normalize any supported input into a PIL.Image, converted to RGB
        (CLIP expects 3-channel input; source images may be grayscale, RGBA,
        palette-mode, etc.).
        """
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            return Image.open(io.BytesIO(image)).convert("RGB")
        raise TypeError(
            f"Unsupported image input type: {type(image).__name__}. "
            "Expected a file path, raw bytes, or a PIL.Image."
        )

    def embed(self, image: ImageInput) -> np.ndarray:
        """
        Generate a 1D normalized float32 embedding vector for a single image.

        Args:
            image: A file path, raw image bytes, or a PIL.Image.

        Returns:
            1D numpy array of shape (dimension,) with dtype float32, L2-normalized.
        """
        pil_image = self._load_image(image)
        embedding = self.model.encode(
            pil_image,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)

    def embed_batch(self, images: List[ImageInput]) -> np.ndarray:
        """
        Generate a 2D normalized float32 embedding matrix for a list of images.

        Args:
            images: List of file paths, raw image bytes, and/or PIL.Images
                (may be mixed).

        Returns:
            2D numpy array of shape (len(images), dimension) with dtype float32, L2-normalized.
        """
        if not images:
            raise ValueError("Input image list for batch embedding cannot be empty.")

        pil_images = [self._load_image(img) for img in images]
        embeddings = self.model.encode(
            pil_images,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
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
