import os
import json
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer


class HybridVectorStore:
    def __init__(self,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 model_cache_dir: str = "./models/embeddings",
                 index_file: str = "./models/index/index_files/faiss_index",
                 metadata_file: str = "./models/index/index_files/metadata_passages.json"):

        self.model_path = os.path.join(model_cache_dir, model_name)
        os.makedirs(self.model_path, exist_ok=True)

        self.model = SentenceTransformer(model_name, cache_folder=self.model_path)
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.index = None
        self.passages = []

    def build_index(self, passages: list):
        """
        Build and save a FAISS index from a list of structured passages.
        Each passage must include a 'text' field (and ideally: page, section, etc.)
        """
        print(f"📦 Received {len(passages)} passages for indexing.")
        texts = [p["text"] for p in passages]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = normalize(embeddings, axis=1)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.passages = passages

        self.save_index()
        print("✅ FAISS index and metadata saved.")

    def save_index(self):
        """Save FAISS index and enriched metadata (JSON format)."""
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        faiss.write_index(self.index, self.index_file)

        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.passages, f, ensure_ascii=False, indent=2)

    def load_index(self):
        """Load a FAISS index and associated passage metadata."""
        if not os.path.exists(self.index_file) or not os.path.exists(self.metadata_file):
            raise FileNotFoundError("❌ FAISS index or metadata JSON not found.")

        self.index = faiss.read_index(self.index_file)
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            self.passages = json.load(f)
        print(f"📂 Loaded index and {len(self.passages)} passages.")

    def search(self, query: str, top_k: int = 5):
        """Semantic search with FAISS and return top_k passages."""
        if self.index is None:
            raise RuntimeError("⚠️ FAISS index is not loaded. Run load_index() or build_index() first.")

        query_embedding = self.model.encode(query, convert_to_numpy=True)
        query_embedding = normalize([query_embedding], axis=1)
        D, I = self.index.search(query_embedding, top_k)

        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.passages):
                results.append(self.passages[idx])
        return results

