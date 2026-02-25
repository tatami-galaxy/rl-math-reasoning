from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Base class for sentence embedders. Plug in any implementation."""

    @abstractmethod
    def embed(self, sentences: list[str]) -> np.ndarray:
        """Embed a flat list of sentences.

        Args:
            sentences: list of N strings.

        Returns:
            Array of shape (N, D).
        """
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Output embedding dimension."""
        ...


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by the sentence-transformers library."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        batch_size: int = 256,
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, sentences: list[str]) -> np.ndarray:
        return self.model.encode(
            sentences,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
