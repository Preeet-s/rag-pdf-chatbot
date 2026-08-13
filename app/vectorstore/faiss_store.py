from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np

from app.ingestion.document import Document


class FAISSVectorStore:
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: list[Document] = []

    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        vectors = np.array(embeddings, dtype="float32")
        self.index.add(vectors)
        self.documents.extend(documents)

    def search(
        self,
        query_embedding: list[float],
        k: int = 3,
    ) -> list[tuple[Document, float]]:
        query = np.array([query_embedding], dtype="float32")

        scores, indices = self.index.search(query, k)

        results: list[tuple[Document, float]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append((self.documents[idx], float(score)))

        return results

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)

        index_path = directory / "index.faiss"
        docs_path = directory / "documents.pkl"

        faiss.write_index(self.index, str(index_path))

        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, directory: Path) -> bool:
        index_path = directory / "index.faiss"
        docs_path = directory / "documents.pkl"

        if not index_path.exists() or not docs_path.exists():
            return False

        self.index = faiss.read_index(str(index_path))

        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)

        return True