import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class VectorStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "faiss.index"
        self.meta_path = self.root / "meta.pkl"
        self.model = SentenceTransformer(settings.embedding_model)
        self.index = faiss.IndexFlatIP(settings.vector_dim)
        self.ids: list[str] = []
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.ids = pickle.loads(self.meta_path.read_bytes())

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
        return vectors / norms

    def add_texts(self, doc_ids: list[str], texts: list[str]) -> None:
        if not texts:
            return
        embeddings = self.model.encode(texts, convert_to_numpy=True).astype(np.float32)
        embeddings = self._normalize(embeddings)
        self.index.add(embeddings)
        self.ids.extend(doc_ids)
        self.persist()

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        query_vec = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        query_vec = self._normalize(query_vec)
        scores, indices = self.index.search(query_vec, top_k)
        result: list[tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            result.append((self.ids[idx], float(score)))
        return result

    def persist(self) -> None:
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_bytes(pickle.dumps(self.ids))
