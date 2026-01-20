"""
Configuration module for Sinhala Letter RAG System.
Provides centralized settings and feature flags for baseline comparisons.
"""

import os
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum


class VectorStoreType(Enum):
    """Supported vector store types."""
    FAISS = "faiss"
    CHROMA = "chroma"


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    OLLAMA = "ollama"  # Local models via Ollama
    HUGGINGFACE = "huggingface"  # Local HuggingFace models


@dataclass
class RetrievalConfig:
    """Configuration for retrieval pipeline."""
    
    # Feature flags for baseline comparison
    use_sinhala_query_builder: bool = True  # Use Sinhala-aware query construction
    use_reranker: bool = True  # Enable cross-encoder reranking (Phase 2)
    
    # Retrieval parameters
    initial_retrieval_k: int = 20  # Candidates for reranker
    final_top_k: int = 3  # Final results after reranking
    
    # Document type filtering
    retrieve_templates: bool = True  # Retrieve structure templates
    retrieve_examples: bool = True  # Retrieve full letter examples
    num_templates: int = 2  # Number of templates to retrieve
    num_examples: int = 2  # Number of examples to retrieve


@dataclass
class EmbeddingConfig:
    """Configuration for embedding model."""
    model_name: str = "sentence-transformers/LaBSE"
    device: str = "cpu"  # Use "cuda" for GPU


@dataclass
class RerankerConfig:
    """Configuration for cross-encoder reranker."""
    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Multilingual fallback
    # For fine-tuned model:
    # model_path: str = "models/reranker/best_model"
    model_path: Optional[str] = None
    device: str = "cpu"
    batch_size: int = 32


@dataclass
class LLMConfig:
    """Configuration for LLM generation."""
    provider: LLMProvider = LLMProvider.OLLAMA  # Changed default to Ollama (free!)
    
    # OpenAI settings
    openai_model: str = "gpt-4"
    openai_temperature: float = 0.3
    
    # Azure OpenAI settings (for production)
    azure_endpoint: Optional[str] = None
    azure_deployment_name: Optional[str] = None
    azure_api_version: str = "2024-02-15-preview"
    
    # Ollama settings (local, free)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "aya:8b"  # Aya 8B - trained on 101 languages including Sinhala
    ollama_temperature: float = 0.3
    
    # HuggingFace settings (local models)
    huggingface_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    huggingface_temperature: float = 0.3
    
    def __post_init__(self):
        """Load Azure settings from environment if available."""
        if self.azure_endpoint is None:
            self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if self.azure_deployment_name is None:
            self.azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


@dataclass
class DataConfig:
    """Configuration for data and vector store."""
    base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))
    
    # Data paths - relative to project root
    data_dir: str = "data"  # Data folder relative to base_dir parent
    csv_filename: str = "sinhala_letters.csv"
    csv_v2_filename: str = "sinhala_letters_v2.csv"  # New schema
    
    # Vector store settings
    vector_store_type: VectorStoreType = VectorStoreType.FAISS
    chroma_persist_dir: str = "chroma_db"
    faiss_index_dir: str = "faiss_index"
    
    @property
    def project_root(self) -> str:
        """Get project root directory (parent of rag/)."""
        return os.path.dirname(self.base_dir)
    
    @property
    def data_path(self) -> str:
        """Get full path to data directory."""
        return os.path.join(self.project_root, self.data_dir)
    
    @property
    def csv_path(self) -> str:
        """Get full path to CSV data file."""
        # First check in data/ folder (preferred location for v2)
        v2_path_data = os.path.join(self.data_path, self.csv_v2_filename)
        if os.path.exists(v2_path_data):
            return v2_path_data
        
        # Check v2 in rag/ folder (legacy)
        v2_path_rag = os.path.join(self.base_dir, self.csv_v2_filename)
        if os.path.exists(v2_path_rag):
            return v2_path_rag
        
        # Check v1 in data/ folder
        v1_path_data = os.path.join(self.data_path, self.csv_filename)
        if os.path.exists(v1_path_data):
            return v1_path_data
        
        # Fallback to v1 in rag/ folder (original location)
        return os.path.join(self.base_dir, self.csv_filename)
    
    @property
    def chroma_path(self) -> str:
        """Get full path to Chroma persist directory."""
        return os.path.join(self.base_dir, self.chroma_persist_dir)
    
    @property
    def faiss_path(self) -> str:
        """Get full path to FAISS index directory."""
        return os.path.join(self.base_dir, self.faiss_index_dir)


@dataclass 
class NERConfig:
    """Configuration for NER model."""
    model_name: str = "xlm-roberta-base"
    use_spacy: bool = True
    use_rules: bool = True
    fine_tuned_model_path: str = "models/training_data/best_model"
    # If True, skip NER and use LLM-based extraction directly (more robust but slower)
    prefer_llm_extraction: bool = True  # Set to True until NER model is trained


@dataclass
class RAGConfig:
    """Main configuration combining all settings."""
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    ner: NERConfig = field(default_factory=NERConfig)
    
    # Experiment tracking
    experiment_name: str = "baseline"
    log_retrieval_results: bool = True


# Global configuration instance
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = RAGConfig()
    return _config


def set_config(config: RAGConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def create_baseline_config() -> RAGConfig:
    """Create configuration for baseline experiments (no enhancements)."""
    config = RAGConfig(
        experiment_name="baseline_labse_only"
    )
    config.retrieval.use_sinhala_query_builder = False
    config.retrieval.use_reranker = False
    return config


def create_sinhala_query_config() -> RAGConfig:
    """Create configuration with Sinhala query builder enabled."""
    config = RAGConfig(
        experiment_name="sinhala_query_builder"
    )
    config.retrieval.use_sinhala_query_builder = True
    config.retrieval.use_reranker = False
    return config


def create_full_pipeline_config() -> RAGConfig:
    """Create configuration with all enhancements enabled."""
    config = RAGConfig(
        experiment_name="full_pipeline_with_reranker"
    )
    config.retrieval.use_sinhala_query_builder = True
    config.retrieval.use_reranker = True
    config.reranker.enabled = True
    return config


# Letter category constants (for validation)
LETTER_CATEGORIES = [
    "request",      # ඉල්ලීම
    "apology",      # ක්ෂමාව
    "invitation",   # ආරාධනා
    "complaint",    # පැමිණිලි
    "application",  # අයදුම්පත්
    "general",      # සාමාන්‍ය
    "notification", # දැනුම්දීම
    "appreciation", # ස්තුති
]

DOC_TYPES = [
    "example",           # Complete letter example
    "structure",         # Letter structure template
    "section_template",  # Reusable section (greeting, closing, etc.)
]

REGISTER_TYPES = [
    "formal",       # Standard formal
    "very_formal",  # Highly formal (government, legal)
]
