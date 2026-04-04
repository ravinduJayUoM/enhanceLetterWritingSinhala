"""
LetterGenerator — Step 5 (final step) of the pipeline.

Accepts an enhanced prompt string and calls the LLM to produce the Sinhala letter.
"""

from typing import Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class LetterGenerator:
    """Generates a formal Sinhala letter by invoking the configured LLM."""

    def __init__(self, llm):
        self.llm = llm
        _prompt = ChatPromptTemplate.from_template("{enhanced_prompt}")
        self._chain = _prompt | self.llm | StrOutputParser()

    def generate(self, enhanced_prompt: str, sender_info: Optional[Dict[str, str]] = None) -> str:
        """Invoke the LLM with *enhanced_prompt* and return the generated letter text."""
        if sender_info:
            sender_block = self._build_sender_block(sender_info)
            enhanced_prompt = enhanced_prompt + "\n\n" + sender_block

        print(f"[LetterGenerator] Prompt length: {len(enhanced_prompt)} chars")
        letter = self._chain.invoke({"enhanced_prompt": enhanced_prompt})
        print(f"[LetterGenerator] Generated {len(letter)} chars")
        return letter

    def _build_sender_block(self, sender_info: Dict[str, str]) -> str:
        """Build a sender info instruction to append to the prompt."""
        lines = ["The letter must include the following sender details in the 'From' section at the top:"]
        if sender_info.get("full_name"):
            lines.append(f"  Name: {sender_info['full_name']}")
        if sender_info.get("title"):
            lines.append(f"  Title/Position: {sender_info['title']}")
        if sender_info.get("address_line1"):
            lines.append(f"  Address: {sender_info['address_line1']}")
        if sender_info.get("address_line2"):
            lines.append(f"           {sender_info['address_line2']}")
        if sender_info.get("phone"):
            lines.append(f"  Phone: {sender_info['phone']}")
        return "\n".join(lines)
