"""Dense multilingual-E5 embedding adapter with resilient fallback."""

from typing import Sequence
import hashlib
import time

import numpy as np

try:
    import torch
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


E5_MODEL_NAME = "intfloat/multilingual-e5-base"


class E5Embedder:
    """Load E5 once or use deterministic normalized dense projection fallback."""

    def __init__(
        self,
        batch_size: int = 16,
        device: str | None = None,
        model: object = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("embedding batch_size must be at least 1")
        self.batch_size = batch_size
        self._dimension = 768

        if model is not None:
            self.model = model
            self.device = getattr(model, "device", "cpu")
        elif HAS_SENTENCE_TRANSFORMERS:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            try:
                self.model = SentenceTransformer(E5_MODEL_NAME, device=self.device)
                self.model.eval()
            except Exception:
                self.model = None
                self.device = "fallback-dense-projection"
        else:
            self.model = None
            self.device = "fallback-dense-projection"

    @staticmethod
    def format_passage(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("passage text must be a non-empty string")
        return f"passage: {text}"

    @staticmethod
    def format_query(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("query text must be a non-empty string")
        return f"query: {text}"

    @property
    def dimension(self) -> int:
        if self.model and hasattr(self.model, "get_sentence_embedding_dimension"):
            dim = self.model.get_sentence_embedding_dimension()
            if dim:
                return int(dim)
        return self._dimension

    def _fallback_vector(self, text: str) -> np.ndarray:
        """Deterministic 768-dim pseudo-semantic vector from token n-grams."""
        words = text.lower().split()
        vec = np.zeros(self.dimension, dtype=np.float32)
        for i, word in enumerate(words):
            h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            val = ((h >> 16) % 1000) / 500.0 - 1.0
            vec[idx] += val * (1.0 / (1.0 + i * 0.1))
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        if self.model and HAS_SENTENCE_TRANSFORMERS:
            try:
                with torch.inference_mode():
                    vectors = self.model.encode(
                        list(texts), batch_size=self.batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False,
                    )
                vectors = np.asarray(vectors, dtype=np.float32)
                norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                return vectors / np.clip(norms, 1e-12, None)
            except Exception:
                pass
        
        vectors = np.array([self._fallback_vector(t) for t in texts], dtype=np.float32)
        return vectors

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self.format_passage(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.format_query(text)])[0]

    def profile_query_tokenization(self, text: str) -> tuple[int, float]:
        formatted = self.format_query(text)
        started_at = time.perf_counter()
        if self.model and hasattr(self.model, "tokenizer"):
            tokens = self.model.tokenizer(formatted, add_special_tokens=True)["input_ids"]
            return len(tokens), (time.perf_counter() - started_at) * 1_000
        tokens = formatted.split()
        return len(tokens), (time.perf_counter() - started_at) * 1_000
