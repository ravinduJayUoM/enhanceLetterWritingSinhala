"""
LetterDatabase — manages loading the CSV knowledge base and the FAISS vector store.

Responsibilities:
  - Load and parse sinhala_letters_v2.csv (v2 schema) or legacy v1 CSV
  - Create LangChain Document objects with proper metadata
  - Build or load the FAISS vector index
  - Expose a search() method used by the Retriever step
"""

import os
import sys
import time
import stat
import shutil
import pandas as pd
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Ensure rag/ is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from knowledge_base.llm_factory import get_embeddings

_config = get_config()
_RAG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # rag/


def _ensure_writable(dir_path: str) -> None:
    """Attempt to make a directory writable; raise PermissionError on failure."""
    try:
        test = os.path.join(dir_path, ".write_test")
        with open(test, "w") as f:
            f.write("")
        os.remove(test)
    except (PermissionError, OSError):
        for root, dirs, files in os.walk(dir_path):
            os.chmod(root, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            for d in dirs:
                os.chmod(os.path.join(root, d), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            for f in files:
                os.chmod(os.path.join(root, f), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


class LetterDatabase:
    """Manages the Sinhala letter knowledge base (CSV → FAISS vector store)."""

    def __init__(self, csv_path: str = None, faiss_path: str = None):
        self.csv_path = csv_path or _config.data.csv_path
        self.faiss_path = faiss_path or _config.data.faiss_path
        self.db: FAISS = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(df)} letters from {self.csv_path}")
        return df

    def create_documents(self, df: pd.DataFrame) -> List[Document]:
        """Convert dataframe rows to LangChain Documents (supports v1 and v2 schema)."""
        documents = []
        columns = set(df.columns)
        is_v2 = "letter_category" in columns and "doc_type" in columns

        print(f"Schema detected: {'v2' if is_v2 else 'v1'}")

        for _, row in df.iterrows():
            if is_v2:
                title = str(row.get("title", ""))
                content = str(row.get("content", ""))
                metadata = {
                    "id": str(row.get("id", "")),
                    "title": title,
                    "letter_category": str(row.get("letter_category", "general")),
                    "doc_type": str(row.get("doc_type", "example")),
                    "register": str(row.get("register", "formal")),
                    "source": str(row.get("source", "curated")),
                    "tags": str(row.get("tags", "")),
                }
            else:
                subject = str(row.get("subject", ""))
                content = str(row.get("content", ""))
                title = subject
                metadata = {
                    "subject": subject,
                    "tags": str(row.get("tags", "")),
                    "source": "sinhala_letters_dataset",
                    "letter_category": "general",
                    "doc_type": "example",
                    "register": "formal",
                }

            documents.append(
                Document(page_content=f"{title}\n\n{content}", metadata=metadata)
            )

        print(f"Created {len(documents)} documents.")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        split = self.text_splitter.split_documents(documents)
        print(f"Split into {len(split)} chunks.")
        return split

    # ------------------------------------------------------------------
    # Vector store management
    # ------------------------------------------------------------------

    def _save_faiss(self, db: FAISS) -> None:
        os.makedirs(self.faiss_path, exist_ok=True)
        db.save_local(self.faiss_path)
        print(f"FAISS index saved to {self.faiss_path}")

    def _load_faiss(self) -> FAISS:
        db = FAISS.load_local(
            self.faiss_path,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        print(f"FAISS index loaded from {self.faiss_path}")
        return db

    def build_knowledge_base(self, force_rebuild: bool = False) -> FAISS:
        """Load existing index or build a new one from the CSV."""
        if not force_rebuild and os.path.exists(self.faiss_path):
            self.db = self._load_faiss()
            return self.db

        print("Building knowledge base from CSV…")
        df = self.load_data()
        docs = self.create_documents(df)
        chunks = self.split_documents(docs)
        self.db = FAISS.from_documents(chunks, get_embeddings())
        self._save_faiss(self.db)
        return self.db

    def rebuild_knowledge_base(self) -> Dict[str, Any]:
        """Force-rebuild and return stats (used by the API endpoint)."""
        start = time.time()
        df = self.load_data()
        docs = self.create_documents(df)
        chunks = self.split_documents(docs)
        self.db = FAISS.from_documents(chunks, get_embeddings())
        self._save_faiss(self.db)
        return {
            "csv_rows": len(df),
            "documents": len(docs),
            "chunks": len(chunks),
            "index_size": len(self.db.index_to_docstore_id),
            "elapsed_seconds": round(time.time() - start, 2),
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        if self.db is None:
            raise RuntimeError("Knowledge base not built. Call build_knowledge_base() first.")
        results = self.db.similarity_search(query, k=top_k)
        print(f"[LetterDatabase] '{query}' → {len(results)} results")
        return results

    def search_filtered(
        self,
        query: str,
        letter_category: str = None,
        doc_type: str = None,
        top_k: int = 5,
        fetch_k: int = 40,
    ) -> List[Document]:
        """Vector search then filter results by metadata fields.

        Fetches *fetch_k* candidates from FAISS, then filters by
        letter_category and/or doc_type, returning up to *top_k* matches.
        """
        if self.db is None:
            raise RuntimeError("Knowledge base not built. Call build_knowledge_base() first.")
        candidates = self.db.similarity_search(query, k=fetch_k)
        filtered = [
            doc for doc in candidates
            if (letter_category is None or doc.metadata.get("letter_category") == letter_category)
            and (doc_type is None or doc.metadata.get("doc_type") == doc_type)
        ]
        print(f"[LetterDatabase] filtered({letter_category},{doc_type}) → {len(filtered)} / {len(candidates)}")
        return filtered[:top_k]

    def add_document(self, content: str, metadata: dict) -> int:
        """Embed a single document and add it directly to the live FAISS index.

        The index is persisted to disk immediately so it survives restarts.
        Returns the number of vectors now in the index.
        """
        if self.db is None:
            raise RuntimeError("Knowledge base not built. Call build_knowledge_base() first.")
        doc = Document(page_content=content, metadata=metadata)
        chunks = self.split_documents([doc])
        self.db.add_documents(chunks)
        self._save_faiss(self.db)
        size = len(self.db.index_to_docstore_id)
        print(f"[LetterDatabase] Added {len(chunks)} chunk(s). Index now has {size} vectors.")
        return size

    # ------------------------------------------------------------------
    # Diagnostics helpers
    # ------------------------------------------------------------------

    def document_count(self) -> int:
        if self.db is None:
            return 0
        return len(self.db.index_to_docstore_id)

    def sample_documents(self, count: int = 3) -> List[Dict[str, Any]]:
        if self.db is None:
            return []
        ids = list(self.db.index_to_docstore_id.values())[:count]
        return [
            {
                "id": doc_id,
                "text": self.db.docstore.search(doc_id).page_content,
                "metadata": self.db.docstore.search(doc_id).metadata,
            }
            for doc_id in ids
        ]
