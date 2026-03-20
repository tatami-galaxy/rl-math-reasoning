from abc import ABC, abstractmethod

import numpy as np
import torch


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


class QwenEmbedder(Embedder):
    """Embedder backed by Qwen3-Embedding models via transformers."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cuda",
        batch_size: int = 256,
        max_seq_len: int = 512,
        instruction: str = "Embed this mathematical reasoning segment",
    ):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.instruction = instruction

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self._dim = self.model.config.hidden_size

    @property
    def dim(self) -> int:
        return self._dim

    def _last_token_pool(self, last_hidden_states, attention_mask):
        """Pool by taking the last non-padding token's hidden state."""
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths,
        ]

    def _format_texts(self, sentences: list[str]) -> list[str]:
        """Prepend instruction prefix if set."""
        if self.instruction:
            return [
                f"Instruct: {self.instruction}\nQuery: {s}" for s in sentences
            ]
        return sentences

    @torch.no_grad()
    def embed(self, sentences: list[str]) -> np.ndarray:
        formatted = self._format_texts(sentences)
        parts = []
        for i in range(0, len(formatted), self.batch_size):
            batch = formatted[i : i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            emb = self._last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            parts.append(emb.cpu().float().numpy())
            print(f"  encoded {min(i + self.batch_size, len(sentences))}/{len(sentences)}", end="\r")
        print()
        return np.concatenate(parts, axis=0)


class SonarEmbedder(Embedder):
    """Embedder backed by Meta's SONAR model. Supports encoding and decoding."""

    def __init__(
        self,
        device: str = "cuda",
        batch_size: int = 256,
        source_lang: str = "eng_Latn",
        target_lang: str = "eng_Latn",
        max_seq_len: int = 512,
        dtype: torch.dtype = torch.float32,
    ):
        from sonar.inference_pipelines.text import (
            TextToEmbeddingModelPipeline,
            EmbeddingToTextModelPipeline,
        )

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_seq_len = max_seq_len

        self.encoder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=self.device,
            dtype=dtype,
        )
        self.decoder = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_encoder",
            device=self.device,
            dtype=dtype,
        )
        # SONAR basic encoder produces 1024-dim embeddings
        self._dim = 1024

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, sentences: list[str]) -> np.ndarray:
        parts = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            emb = self.encoder.predict(batch, source_lang=self.source_lang)
            parts.append(emb.cpu().float().numpy())
            print(f"  encoded {min(i + self.batch_size, len(sentences))}/{len(sentences)}", end="\r")
        print()
        return np.concatenate(parts, axis=0)

    def decode(self, embeddings: np.ndarray, sampler=None) -> list[str]:
        """Decode embeddings back to text.

        Args:
            embeddings: Array of shape (N, D).
            sampler: Optional fairseq2 sampler (e.g. TopPSampler, TopKSampler).
                     If None, uses default beam search.

        Returns:
            List of N decoded strings.
        """
        emb_tensor = torch.from_numpy(embeddings).to(self.device)
        kwargs = dict(target_lang=self.target_lang, max_seq_len=self.max_seq_len)
        if sampler is not None:
            kwargs["sampler"] = sampler
        decoded = []
        for i in range(0, len(emb_tensor), self.batch_size):
            batch = emb_tensor[i : i + self.batch_size]
            decoded.extend(self.decoder.predict(batch, **kwargs))
            print(f"  decoded {min(i + self.batch_size, len(emb_tensor))}/{len(emb_tensor)}", end="\r")
        print()
        return decoded
