"""
LetterGenerator — Step 5 (final step) of the pipeline.

Accepts an enhanced prompt string and calls the LLM to produce the Sinhala letter.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class LetterGenerator:
    """Generates a formal Sinhala letter by invoking the configured LLM."""

    def __init__(self, llm):
        self.llm = llm
        _prompt = ChatPromptTemplate.from_template("{enhanced_prompt}")
        self._chain = _prompt | self.llm | StrOutputParser()

    def generate(self, enhanced_prompt: str) -> str:
        """Invoke the LLM with *enhanced_prompt* and return the generated letter text."""
        print(f"[LetterGenerator] Prompt length: {len(enhanced_prompt)} chars")
        letter = self._chain.invoke({"enhanced_prompt": enhanced_prompt})
        print(f"[LetterGenerator] Generated {len(letter)} chars")
        return letter
