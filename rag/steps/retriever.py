"""
Retriever — Step 3 of the pipeline.

Converts extracted letter metadata into a search query (using SinhalaQueryBuilder),
queries the FAISS vector store, and optionally reranks results with a cross-encoder.
"""

import os
import sys
from typing import Any, Dict, List

from langchain_core.documents import Document

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from query_builder import SinhalaQueryBuilder
from reranker import CrossEncoderReranker


class Retriever:
    """Retrieves relevant letter examples from the knowledge base."""

    def __init__(self, letter_database, config=None):
        """
        Args:
            letter_database: An initialised LetterDatabase instance.
            config: Optional RAGConfig override; defaults to the global config.
        """
        self.db = letter_database
        self.config = config or get_config()
        self.query_builder = SinhalaQueryBuilder()

        retrieval = self.config.retrieval
        if retrieval.use_reranker:
            self.reranker = CrossEncoderReranker(
                model_name_or_path=self.config.reranker.model_name,
                device=self.config.reranker.device,
            )
        else:
            self.reranker = None

    def retrieve(self, extracted_info: Dict[str, Any], top_k: int = None) -> List[Document]:
        """
        Build a query from *extracted_info* and retrieve documents.

        Strategy:
          1. Fetch 1 structure template matching the letter_category.
          2. Fetch 2 examples matching the letter_category (reranked if enabled).
          3. Fall back to unfiltered reranked results if either bucket is empty.

        Returns the combined list (structure first, then examples).
        """
        retrieval = self.config.retrieval

        if retrieval.use_sinhala_query_builder:
            query = self.query_builder.build_query(extracted_info)
            print(f"[Retriever] Sinhala query: {query}")
        else:
            parts = [
                extracted_info.get("letter_type", ""),
                extracted_info.get("subject", ""),
                extracted_info.get("purpose", ""),
                extracted_info.get("details", ""),
            ]
            query = " ".join(p for p in parts if p)
            print(f"[Retriever] Legacy query: {query}")

        category = extracted_info.get("letter_type", "") or None

        # --- Bucket 1: structure template for this category ---
        structures = self.db.search_filtered(
            query, letter_category=category, doc_type="structure", top_k=1, fetch_k=40
        )
        # Fallback: any structure if none found for this category
        if not structures:
            structures = self.db.search_filtered(
                query, letter_category=None, doc_type="structure", top_k=1, fetch_k=40
            )

        # --- Bucket 2: examples for this category (reranked) ---
        example_candidates = self.db.search_filtered(
            query, letter_category=category, doc_type="example", top_k=10, fetch_k=40
        )
        # Fallback: any examples if none found for this category
        if not example_candidates:
            example_candidates = self.db.search_filtered(
                query, letter_category=None, doc_type="example", top_k=10, fetch_k=40
            )

        if self.reranker and example_candidates:
            doc_dicts = [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in example_candidates
            ]
            reranked = self.reranker.rerank(query, doc_dicts, top_k=2)
            examples = [
                Document(page_content=d["content"], metadata=d["metadata"])
                for d in reranked
            ]
        else:
            examples = example_candidates[:2]

        results = structures + examples
        print(f"[Retriever] Final: {len(structures)} structure(s) + {len(examples)} example(s)")

        if self.config.log_retrieval_results:
            for i, doc in enumerate(results):
                cat = doc.metadata.get("letter_category", "?")
                dt = doc.metadata.get("doc_type", "?")
                print(f"  [{i+1}] category={cat}, doc_type={dt}")

        return results
