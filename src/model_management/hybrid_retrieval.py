import numpy as np
import faiss
import json
import re
from nltk.tokenize import RegexpTokenizer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class HybridRetriever:
    def __init__(self, index_path: str, embeddings_path: str, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

        # Loading FAISS
        try:
            self.index = faiss.read_index(index_path)
        except Exception as e:
            raise RuntimeError(f"Error loading index FAISS : {e}")

        # Loading passages
        try:
            with open(embeddings_path, 'r', encoding='utf-8') as f:
                self.passages = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading indexed passages : {e}")

        # tokenizer
        self.tokenizer = RegexpTokenizer(r"\w+")
        self.corpus = [self.tokenizer.tokenize(p['text'].lower()) for p in self.passages]
        self.bm25 = BM25Okapi(self.corpus)

    def retrieve_hybrid(self, query: str, top_k: int = 5, alpha: float = 0.6) -> str:
        """
        Hybrid semantic (FAISS) + lexical (BM25) search
        alpha : weight of semantic search (between 0 and 1)
        """
        faiss_results = self.search_faiss(query, top_k=top_k * 2)
        bm25_results = self.search_bm25(query, top_k=top_k * 2)

        # Score normalization
        max_faiss = max([s for _, s in faiss_results], default=1e-6)
        faiss_norm = {i: s / max_faiss for i, s in faiss_results}

        max_bm25 = max([s for _, s in bm25_results], default=1e-6)
        bm25_norm = {i: s / max_bm25 for i, s in bm25_results}

        # Weighted merger
        combined_scores = {}
        for idx in set(faiss_norm.keys()).union(bm25_norm.keys()):
            combined_scores[idx] = alpha * faiss_norm.get(idx, 0) + (1 - alpha) * bm25_norm.get(idx, 0)

        # Sorting results
        top_idxs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        passages = [self._format_result(self.passages[i], combined_scores[i]) for i, _ in top_idxs]
        return "\n\n".join(passages) if passages else "Aucune information pertinente trouvée."

    def search_faiss(self, query: str, top_k: int = 10):
        embedding = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(np.array([embedding]), top_k)
        return [(idx, float(score)) for score, idx in zip(D[0], I[0]) if idx != -1]

    def search_bm25(self, query: str, top_k: int = 10):
        tokenized_query = self.tokenizer.tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(idx, float(scores[idx])) for idx in ranked]

    def _format_result(self, passage, score):
        document = passage.get('document', 'Unspecified')
        section = passage.get('section', 'Unspecified')
        subsection = passage.get('subsection', 'Unspecified')
        page = passage.get('page', 'Unspecified')
        text = passage.get('text', '')
        return f"Document: {document}\nPage: {page}\nSection: {section}\nSubsection: {subsection}\nText: {text}\nCombined score: {score:.4f}"

    def highlight_keywords(self, text: str, query: str) -> str:
        for word in self.tokenizer.tokenize(query):
            pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
            text = pattern.sub(r"\033[1m\1\033[0m", text)
        return text





