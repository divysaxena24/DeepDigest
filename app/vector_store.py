import os
import json
import faiss
import numpy as np
from typing import List, Tuple

class FAISSVectorStore:
    """
    Persistent FAISS vector index with normalized cosine similarity (IndexFlatIP)
    and JSON metadata sidecar persistence.
    """
    def __init__(self, dimension: int = 384, storage_dir: str = "data_store"):
        self.dimension = dimension
        self.storage_dir = storage_dir
        self.index_path = os.path.join(storage_dir, "deepdigest.index")
        self.meta_path = os.path.join(storage_dir, "metadata.json")
        self.chunks: List[str] = []
        
        os.makedirs(self.storage_dir, exist_ok=True)
        self._initialize_index()

    def _initialize_index(self):
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self.load()
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunks = []

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        vectors = vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms

    def add_chunks(self, chunks: List[str], embeddings: np.ndarray):
        if len(chunks) == 0:
            return
        
        normed = self._normalize(embeddings)
        self.index.add(normed)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 1) -> List[Tuple[str, float]]:
        if self.index.ntotal == 0 or len(self.chunks) == 0:
            return []

        normed_q = self._normalize(query_embedding.reshape(1, -1))
        scores, indices = self.index.search(normed_q, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results

    def reset(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.meta_path):
            os.remove(self.meta_path)

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"chunks": self.chunks}, f, ensure_ascii=False)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.chunks = data.get("chunks", [])

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal
