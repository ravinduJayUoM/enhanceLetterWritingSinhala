"""
PromptBuilder — Step 4 of the pipeline.

Combines extracted letter metadata, retrieved example documents, and any
user-supplied answers to gap-filling questions into a single enhanced prompt
ready to be passed to the letter generation LLM.

Pure logic — no LLM or I/O dependencies.
"""

from datetime import date
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


class PromptBuilder:
    """Constructs an enhanced generation prompt from pipeline artefacts."""

    def build(
        self,
        original_prompt: str,
        extracted_info: Dict[str, Any],
        retrieved_docs: List[Document],
        missing_info_answers: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Args:
            original_prompt:      The raw user prompt string.
            extracted_info:       Output of InfoExtractor.extract().
            retrieved_docs:       Output of Retriever.retrieve().
            missing_info_answers: Optional answers collected by GapFiller.

        Returns:
            A fully-formed prompt string to send to LetterGenerator.
        """
        # Merge extracted info with any gap-filler answers
        info = dict(extracted_info)
        if missing_info_answers:
            info.update(missing_info_answers)

        today = date.today().strftime("%Y %B %d")  # e.g. "2026 April 04"

        examples_str = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)

        # Build the event details block for invitations
        event_details = ""
        if info.get("letter_type") == "invitation":
            event_parts = []
            if info.get("event_date"):
                event_parts.append(f"- Event Date:  {info['event_date']}")
            if info.get("event_time"):
                event_parts.append(f"- Event Time:  {info['event_time']}")
            if info.get("event_venue"):
                event_parts.append(f"- Event Venue: {info['event_venue']}")
            if event_parts:
                event_details = "\n" + "\n".join(event_parts)

        prompt = f"""You are a Sinhala formal letter writing assistant. \
Generate a complete formal letter IN SINHALA based on the information and examples below.

IMPORTANT: Write the letter ONLY in Sinhala script. \
Do not translate, explain, or include any English text.

Original Request: {original_prompt}

Letter Details:
- Type:               {info.get('letter_type', 'general')}
- Recipient:          {info.get('recipient', '')}
- Sender:             {info.get('sender', '')}
- Subject:            {info.get('subject', '')}
- Purpose:            {info.get('purpose', '')}
- Additional Details: {info.get('details', '')}{event_details}

Reference Letter Examples (use these for structure and formal language):
{examples_str}

Instructions:
1. Write a complete formal letter in Sinhala following the structure of the examples above.
2. Use proper Sinhala grammar, punctuation, and formal register.
3. Include appropriate formal greetings and closings.
4. Address all the details mentioned above.
5. Use today's date ({today}) in the letter unless a specific date is mentioned in the details above.
6. Output ONLY the letter content in Sinhala — no explanations, no English text, no notes.

Generate the letter now:"""

        print("\n" + "="*60)
        print("[PromptBuilder] Final prompt sent to LLM:")
        print("="*60)
        print(prompt)
        print("="*60 + "\n")
        return prompt
