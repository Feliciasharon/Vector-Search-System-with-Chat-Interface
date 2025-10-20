
import numpy as np
from typing import List, Optional, Dict, Any


class VectorEntry:
    """A single record in the vector database."""
    def __init__(self, id: str, vector: np.ndarray, text: str):
        self.id = id
        self.vector = np.asarray(vector, dtype=np.float32)
        self.text = text


class SimpleVectorDB:
    
    def __init__(self):
        self.entries: List[VectorEntry] = []
        self.dim: Optional[int] = None

    def insert(self, id: str, vector: np.ndarray, text: str):
        
        vector = np.asarray(vector, dtype=np.float32)
        if self.dim is None:
            self.dim = vector.shape[0]
        elif vector.shape[0] != self.dim:
            raise ValueError("Vector dimensionality mismatch.")
        self.entries.append(VectorEntry(id, vector, text))

    def _score(self, q: np.ndarray, v: np.ndarray, metric: str) -> float:
        
        if metric == "cosine":
            qn = q / (np.linalg.norm(q) + 1e-12)
            vn = v / (np.linalg.norm(v) + 1e-12)
            return float(np.dot(qn, vn))  
        elif metric == "dot":
            return float(np.dot(q, v))    
        elif metric == "euclidean":
            return -float(np.linalg.norm(q - v))  
        else:
            raise ValueError("Metric must be one of ['cosine', 'dot', 'euclidean'].")

    def search(self, query_vec: np.ndarray, k: int = 5, metric: str = "cosine") -> List[Dict[str, Any]]:
        """Return top-k most similar vectors."""
        if not self.entries:
            return []

        q = np.asarray(query_vec, dtype=np.float32)
        scores = []
        for e in self.entries:
            score = self._score(q, e.vector, metric)
            scores.append((score, e))
        scores.sort(key=lambda x: x[0], reverse=True)

        return [
            {"id": e.id, "score": float(s), "text": e.text}
            for s, e in scores[:k]
        ]

    def __len__(self):
        return len(self.entries)
