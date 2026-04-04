"""
Pipeline — orchestrates all five steps of the Sinhala letter generation pipeline.

  Step 1  InfoExtractor    — NER / LLM based extraction of letter metadata
  Step 2  GapFiller        — detect missing required fields, generate questions
  Step 3  Retriever        — vector search + cross-encoder reranking
  Step 4  PromptBuilder    — template-based prompt optimisation
  Step 5  LetterGenerator  — LLM letter generation

The `process()` method runs Steps 1–4.
The `generate_letter()` method runs Step 5.

Keeping these two separate mirrors the existing API contract where
/process_query/ and /generate_letter/ are distinct endpoints.
"""

from typing import Any, Dict, List, Optional

from steps.extractor import InfoExtractor
from steps.gap_filler import GapFiller
from steps.retriever import Retriever
from steps.prompt_builder import PromptBuilder
from steps.generator import LetterGenerator


class Pipeline:
    def __init__(
        self,
        extractor: InfoExtractor,
        gap_filler: GapFiller,
        retriever: Retriever,
        prompt_builder: PromptBuilder,
        generator: LetterGenerator,
    ):
        self.extractor = extractor
        self.gap_filler = gap_filler
        self.retriever = retriever
        self.prompt_builder = prompt_builder
        self.generator = generator

    def process(
        self,
        prompt: str,
        missing_info: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Run Steps 1–4.

        Returns either:
          {"status": "incomplete", "extracted_info": ..., "missing_fields": ..., "questions": ...}
        or:
          {"status": "complete", "extracted_info": ..., "enhanced_prompt": ..., "relevant_docs": [...]}
        """
        # Step 1 — extract
        extracted = self.extractor.extract(prompt)

        # Step 2 — gap detection (skip if the caller already supplied answers)
        if not missing_info:
            missing_fields = self.gap_filler.identify_missing(extracted)
            if missing_fields:
                questions = self.gap_filler.generate_questions(
                    missing_fields, letter_type=extracted.get("letter_type", "general")
                )
                return {
                    "status": "incomplete",
                    "extracted_info": extracted,
                    "missing_fields": missing_fields,
                    "questions": questions,
                }

        # Step 3 — retrieve
        retrieved_docs = self.retriever.retrieve(extracted)

        # Step 4 — build enhanced prompt
        enhanced_prompt = self.prompt_builder.build(
            prompt, extracted, retrieved_docs, missing_info
        )

        return {
            "status": "complete",
            "extracted_info": extracted,
            "enhanced_prompt": enhanced_prompt,
            "relevant_docs": [doc.page_content for doc in retrieved_docs],
        }

    def generate_letter(self, enhanced_prompt: str, sender_info: Optional[Dict[str, str]] = None) -> str:
        """Step 5 — generate the Sinhala letter from an enhanced prompt."""
        return self.generator.generate(enhanced_prompt, sender_info)
