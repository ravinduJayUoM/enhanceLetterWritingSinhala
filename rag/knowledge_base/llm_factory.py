"""
LLM factory — creates and returns an LLM instance based on the active configuration.
Supports Ollama (default), HuggingFace, Azure OpenAI, and standard OpenAI.
Also provides a shared embeddings singleton to avoid reloading the model on every call.
"""

import os
import sys

# Ensure rag/ is on the path so that config can be imported from step sub-packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config, LLMProvider

_embeddings_singleton = None


def get_embeddings():
    """Return a cached HuggingFaceEmbeddings instance (LaBSE by default)."""
    global _embeddings_singleton
    if _embeddings_singleton is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        cfg = get_config()
        _embeddings_singleton = HuggingFaceEmbeddings(
            model_name=cfg.embedding.model_name,
            model_kwargs={"device": cfg.embedding.device},
        )
    return _embeddings_singleton


def get_llm(temperature: float = 0.3, provider=None):
    """
    Create and return an LLM instance driven by RAGConfig.llm.provider.

    Provider selection:
      GEMINI      — Google Gemini via langchain-google-genai (requires GEMINI_API_KEY)
      OLLAMA      — local model via Ollama
      HUGGINGFACE — local HuggingFace pipeline
      AZURE_OPENAI— Azure-hosted OpenAI
      OPENAI      — standard OpenAI API (fallback)

    To switch provider, change RAGConfig.llm.provider in config.py
    or call set_config() before starting the server.
    To switch models within a provider, update the matching *_model field.
    API keys are always read from environment variables (never hard-coded).

    Args:
        provider: Optional LLMProvider override. If given, uses this provider
                  instead of the one in config. Useful for split-provider setups
                  (e.g. Gemini for extraction, Ollama for generation).
    """
    from langchain_openai import ChatOpenAI, AzureChatOpenAI

    cfg = get_config()
    provider = provider or cfg.llm.provider

    if provider == LLMProvider.GEMINI:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is not installed. "
                "Run: pip install langchain-google-genai"
            )
        api_key = cfg.llm.gemini_api_key
        if not api_key:
            raise ValueError(
                "Gemini API key not found. "
                "Set the GEMINI_API_KEY environment variable."
            )
        print(f"[LLMFactory] Using Gemini: {cfg.llm.gemini_model}")
        return ChatGoogleGenerativeAI(
            model=cfg.llm.gemini_model,
            google_api_key=api_key,
            temperature=temperature,
        )

    if provider == LLMProvider.OLLAMA:
        from langchain_community.llms import Ollama
        print(f"[LLMFactory] Using Ollama: {cfg.llm.ollama_model}")
        return Ollama(
            model=cfg.llm.ollama_model,
            base_url=cfg.llm.ollama_base_url,
            temperature=temperature,
        )

    if provider == LLMProvider.HUGGINGFACE:
        from langchain_community.llms import HuggingFacePipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
        print(f"[LLMFactory] Using HuggingFace: {cfg.llm.huggingface_model}")
        tokenizer = AutoTokenizer.from_pretrained(cfg.llm.huggingface_model)
        model = AutoModelForCausalLM.from_pretrained(cfg.llm.huggingface_model)
        pipe = hf_pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512
        )
        return HuggingFacePipeline(pipeline=pipe)

    if provider == LLMProvider.AZURE_OPENAI:
        azure_endpoint = cfg.llm.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_deployment = (
            cfg.llm.azure_deployment_name
            or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4-deployment")
        )
        if not (azure_endpoint and azure_key):
            raise ValueError(
                "Azure OpenAI credentials missing. "
                "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY."
            )
        print(f"[LLMFactory] Using Azure OpenAI: {azure_deployment}")
        return AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            azure_deployment=azure_deployment,
            api_version=cfg.llm.azure_api_version,
            temperature=temperature,
        )

    # Default: standard OpenAI
    print(f"[LLMFactory] Using OpenAI: {cfg.llm.openai_model}")
    return ChatOpenAI(model=cfg.llm.openai_model, temperature=temperature)
