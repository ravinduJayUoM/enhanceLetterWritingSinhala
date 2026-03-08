"""
GapFiller — Step 2 of the pipeline.

Identifies missing required fields for the target letter type and
generates Sinhala clarifying questions for the user to answer.
Pure logic — no LLM or I/O dependencies.
"""

from typing import Dict, List, Any

_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "application": ["recipient", "sender", "subject", "purpose", "details"],
    "complaint":   ["recipient", "sender", "subject", "purpose", "details"],
    "request":     ["recipient", "sender", "subject", "purpose"],
    "general":     ["recipient", "sender", "subject", "purpose"],
}

_QUESTIONS: Dict[str, str] = {
    "recipient": "ලිපිය යොමු කළ යුත්තේ කාටද?",
    "sender":    "ලිපිය යවන්නේ කවුරුන්ද?",
    "subject":   "ලිපියේ මාතෘකාව කුමක්ද?",
    "purpose":   "ලිපියේ මූලික අරමුණ කුමක්ද?",
    "details":   "ලිපියට අදාළ වැදගත් විස්තර (දිනය, ස්ථානය, ප්‍රමාණය, ආදිය) කුමක්ද?",
}


class GapFiller:
    """Determines which required fields are absent and generates questions for them."""

    def identify_missing(self, extracted_info: Dict[str, Any]) -> List[str]:
        """Return list of required field names that are empty in *extracted_info*."""
        letter_type = extracted_info.get("letter_type", "general")
        required = _REQUIRED_FIELDS.get(letter_type, _REQUIRED_FIELDS["general"])
        return [
            field for field in required
            if not str(extracted_info.get(field, "")).strip()
            or extracted_info.get(field) == "N/A"
        ]

    def generate_questions(self, missing_fields: List[str]) -> Dict[str, str]:
        """Map each missing field to a Sinhala question string."""
        return {
            field: _QUESTIONS.get(field, f"කරුණාකර {field} සපයන්න")
            for field in missing_fields
        }
