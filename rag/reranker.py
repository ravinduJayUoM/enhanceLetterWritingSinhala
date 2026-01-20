"""
Cross-Encoder Reranker for RAG pipeline (Phase 2)
"""
from typing import List, Tuple
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name_or_path: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        self.model = CrossEncoder(model_name_or_path, device=device)

    def rerank(self, query: str, docs: List[dict], top_k: int = 3) -> List[dict]:
        # docs: list of dicts with 'content' or 'text' field
        pairs = [(query, doc.get('content') or doc.get('text', '')) for doc in docs]
        scores = self.model.predict(pairs)
        # Attach scores and sort
        for doc, score in zip(docs, scores):
            doc['rerank_score'] = float(score)
        reranked = sorted(docs, key=lambda d: d['rerank_score'], reverse=True)
        return reranked[:top_k]
