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
    "request":     ["recipient", "sender", "subject", "purpose", "details"],
    "invitation":  ["recipient", "sender", "subject", "purpose", "event_date", "event_time", "event_venue"],
    "general":     ["recipient", "sender", "subject", "purpose"],
}

# Type-specific overrides for the "details" question
_DETAILS_QUESTIONS: Dict[str, str] = {
    "request":   "ඔබගේ ඉල්ලීමට අදාළ කාල සීමාව හෝ දිනය සඳහන් කළ හැකිය (නිවාඩු ගන්නා කාලය, රැස්වීමේ දිනය, ආදිය). නිශ්චිත නොමැති නම් හැකි තරම් විස්තර කරන්න.",
    "complaint": "සිද්ධිය සිදු වූ දිනය, වේලාව හෝ ස්ථානය සඳහන් කළ හැකිය. නිශ්චිත නොදන්නේ නම් ආසන්න කාලය හෝ 'නොදනිමි' ලෙස සඳහන් කරන්න.",
    "default":   "ලිපියට අදාළ වැදගත් විස්තර (දිනය, ස්ථානය, ප්‍රමාණය, ආදිය) කුමක්ද?",
}

_QUESTIONS: Dict[str, str] = {
    "recipient":   "ලිපිය යොමු කළ යුත්තේ කාටද?",
    "sender":      "ලිපිය යවන්නේ කවුරුන්ද?",
    "subject":     "ලිපියේ මාතෘකාව කුමක්ද?",
    "purpose":     "ලිපියේ මූලික අරමුණ කුමක්ද?",
    "details":     _DETAILS_QUESTIONS["default"],
    "event_date":  "අවස්ථාවේ දිනය කුමක්ද? (උදා: 2026 අප්‍රේල් 15)",
    "event_time":  "අවස්ථාව ආරම්භ වන වේලාව කුමක්ද? (උදා: පෙ.ව. 10.00)",
    "event_venue": "අවස්ථාව පැවැත්වෙන ස්ථානය කුමක්ද?",
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

    def generate_questions(self, missing_fields: List[str], letter_type: str = "general") -> Dict[str, str]:
        """Map each missing field to a Sinhala question string."""
        questions = {}
        for field in missing_fields:
            if field == "details":
                questions[field] = _DETAILS_QUESTIONS.get(letter_type, _DETAILS_QUESTIONS["default"])
            else:
                questions[field] = _QUESTIONS.get(field, f"කරුණාකර {field} සපයන්න")
        return questions
