"""
Sinhala-aware Query Builder for improved retrieval.
Constructs search queries using Sinhala terminology for better embedding alignment.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass


# Mapping from English letter types to Sinhala equivalents
LETTER_TYPE_MAPPING = {
    # Primary mappings
    "request": "ඉල්ලීමේ ලිපිය",
    "apology": "ක්ෂමා ලිපිය",
    "invitation": "ආරාධනා ලිපිය",
    "complaint": "පැමිණිලි ලිපිය",
    "application": "අයදුම්පත් ලිපිය",
    "general": "සාමාන්‍ය ලිපිය",
    "notification": "දැනුම්දීමේ ලිපිය",
    "appreciation": "ස්තුති ලිපිය",
    "resignation": "ඉල්ලා අස්වීමේ ලිපිය",
    "leave": "නිවාඩු ඉල්ලීමේ ලිපිය",
    "transfer": "මාරු ඉල්ලීමේ ලිපිය",
    "recommendation": "නිර්දේශ ලිපිය",
    "cover_letter": "ආවරණ ලිපිය",
    
    # Alternative spellings / variations
    "req": "ඉල්ලීමේ ලිපිය",
    "app": "අයදුම්පත් ලිපිය",
    "comp": "පැමිණිලි ලිපිය",
    "inv": "ආරාධනා ලිපිය",
}

# Sinhala keywords for query enrichment
SINHALA_QUERY_KEYWORDS = {
    "subject": "මාතෘකාව",
    "purpose": "අරමුණ",
    "details": "විස්තර",
    "recipient": "ලබන්නා",
    "sender": "යවන්නා",
    "date": "දිනය",
    "salutation": "ආමන්ත්‍රණය",
    "closing": "අවසාන වාක්‍යය",
    "signature": "අත්සන",
    "body": "ලිපි අන්තර්ගතය",
    "formal": "විධිමත්",
    "official": "නිල",
}

# Common Sinhala formal phrases for query enrichment
FORMAL_PHRASES = {
    "greeting": ["ගරු", "මහත්මයාණෙනි", "මහත්මිය", "තුමා", "තුමිය"],
    "closing": ["ස්තුතියි", "ගෞරවයෙන්", "විශ්වාසී", "බැතිමත්"],
    "request_phrases": ["කාරුණිකව ඉල්ලා සිටිමි", "ඉල්ලා සිටිමි", "ඉල්ලමි"],
    "formal_markers": ["විධිමත්", "නිල", "රාජකාරී"],
}


@dataclass
class QueryComponents:
    """Structured components extracted for query building."""
    letter_type: str = ""
    letter_type_si: str = ""
    subject: str = ""
    purpose: str = ""
    details: str = ""
    recipient: str = ""
    sender: str = ""
    category: str = ""
    register: str = "formal"
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "letter_type": self.letter_type,
            "letter_type_si": self.letter_type_si,
            "subject": self.subject,
            "purpose": self.purpose,
            "details": self.details,
            "recipient": self.recipient,
            "sender": self.sender,
            "category": self.category,
            "register": self.register,
        }


class SinhalaQueryBuilder:
    """
    Builds Sinhala-aware search queries for better retrieval.
    
    This improves embedding alignment by:
    1. Mapping English letter types to Sinhala equivalents
    2. Adding Sinhala field markers to the query
    3. Enriching with formal language indicators
    """
    
    def __init__(
        self,
        include_field_markers: bool = True,
        include_formal_markers: bool = True,
        max_query_length: int = 500
    ):
        """
        Initialize the query builder.
        
        Args:
            include_field_markers: Add Sinhala field labels (මාතෘකාව, අරමුණ, etc.)
            include_formal_markers: Add formal language indicators
            max_query_length: Maximum characters for the final query
        """
        self.include_field_markers = include_field_markers
        self.include_formal_markers = include_formal_markers
        self.max_query_length = max_query_length
    
    def map_letter_type(self, letter_type: str) -> str:
        """
        Map English letter type to Sinhala equivalent.
        
        Args:
            letter_type: English letter type (e.g., "request", "complaint")
            
        Returns:
            Sinhala letter type string
        """
        if not letter_type:
            return "සාමාන්‍ය ලිපිය"
        
        # Normalize input
        letter_type_lower = letter_type.lower().strip()
        
        # Check direct mapping
        if letter_type_lower in LETTER_TYPE_MAPPING:
            return LETTER_TYPE_MAPPING[letter_type_lower]
        
        # Check if it's already in Sinhala (contains Sinhala characters)
        if any('\u0D80' <= c <= '\u0DFF' for c in letter_type):
            return letter_type
        
        # Partial match
        for key, value in LETTER_TYPE_MAPPING.items():
            if key in letter_type_lower or letter_type_lower in key:
                return value
        
        # Default
        return "සාමාන්‍ය ලිපිය"
    
    def build_query(
        self,
        extracted_info: Dict[str, Any],
        original_prompt: Optional[str] = None
    ) -> str:
        """
        Build a Sinhala-aware search query from extracted information.
        
        Args:
            extracted_info: Dictionary with extracted fields (letter_type, subject, etc.)
            original_prompt: Original user prompt (for fallback/enrichment)
            
        Returns:
            Optimized search query string
        """
        # Extract components
        letter_type = extracted_info.get("letter_type", "")
        letter_type_si = self.map_letter_type(letter_type)
        subject = extracted_info.get("subject", "")
        purpose = extracted_info.get("purpose", "")
        details = extracted_info.get("details", "")
        recipient = extracted_info.get("recipient", "")
        sender = extracted_info.get("sender", "")
        
        # Build query parts
        query_parts = []
        
        # 1. Add Sinhala letter type (most important for category matching)
        if letter_type_si:
            query_parts.append(letter_type_si)
        
        # 2. Add subject with Sinhala marker
        if subject:
            if self.include_field_markers:
                query_parts.append(f"{SINHALA_QUERY_KEYWORDS['subject']} {subject}")
            else:
                query_parts.append(subject)
        
        # 3. Add purpose with Sinhala marker
        if purpose:
            if self.include_field_markers:
                query_parts.append(f"{SINHALA_QUERY_KEYWORDS['purpose']} {purpose}")
            else:
                query_parts.append(purpose)
        
        # 4. Add details (truncated if too long)
        if details:
            if self.include_field_markers:
                query_parts.append(f"{SINHALA_QUERY_KEYWORDS['details']} {details[:200]}")
            else:
                query_parts.append(details[:200])
        
        # 5. Add recipient/sender if present
        if recipient:
            if self.include_field_markers:
                query_parts.append(f"{SINHALA_QUERY_KEYWORDS['recipient']} {recipient}")
            else:
                query_parts.append(recipient)
        
        if sender:
            if self.include_field_markers:
                query_parts.append(f"{SINHALA_QUERY_KEYWORDS['sender']} {sender}")
            else:
                query_parts.append(sender)
        
        # 6. Add formal marker if enabled
        if self.include_formal_markers:
            query_parts.append(SINHALA_QUERY_KEYWORDS['formal'])
        
        # Combine parts
        query = " ".join(filter(None, query_parts))
        
        # Truncate if too long
        if len(query) > self.max_query_length:
            query = query[:self.max_query_length]
        
        # Fallback to original prompt if query is too short
        if len(query) < 10 and original_prompt:
            query = original_prompt[:self.max_query_length]
        
        return query.strip()
    
    def build_query_simple(
        self,
        letter_type: str = "",
        subject: str = "",
        purpose: str = "",
        details: str = ""
    ) -> str:
        """
        Simplified query builder with direct parameters.
        
        Args:
            letter_type: Type of letter (English or Sinhala)
            subject: Letter subject
            purpose: Letter purpose
            details: Additional details
            
        Returns:
            Optimized search query string
        """
        return self.build_query({
            "letter_type": letter_type,
            "subject": subject,
            "purpose": purpose,
            "details": details,
        })
    
    def build_multi_query(
        self,
        extracted_info: Dict[str, Any],
        num_queries: int = 3
    ) -> List[str]:
        """
        Build multiple query variations for hybrid retrieval.
        
        This can be used to retrieve more diverse results by querying
        with different emphases (type-focused, content-focused, etc.)
        
        Args:
            extracted_info: Dictionary with extracted fields
            num_queries: Number of query variations to generate
            
        Returns:
            List of query strings
        """
        queries = []
        
        letter_type = extracted_info.get("letter_type", "")
        letter_type_si = self.map_letter_type(letter_type)
        subject = extracted_info.get("subject", "")
        purpose = extracted_info.get("purpose", "")
        details = extracted_info.get("details", "")
        
        # Query 1: Type-focused (for category matching)
        if letter_type_si:
            type_query = f"{letter_type_si} {SINHALA_QUERY_KEYWORDS['formal']} ලිපිය"
            queries.append(type_query)
        
        # Query 2: Content-focused (subject + purpose)
        if subject or purpose:
            content_query = f"{subject} {purpose}".strip()
            if content_query:
                queries.append(content_query)
        
        # Query 3: Full combined query
        full_query = self.build_query(extracted_info)
        if full_query and full_query not in queries:
            queries.append(full_query)
        
        # Ensure we have at least one query
        if not queries:
            queries.append("විධිමත් ලිපිය")  # Default: "formal letter"
        
        return queries[:num_queries]
    
    def extract_components(self, extracted_info: Dict[str, Any]) -> QueryComponents:
        """
        Extract and structure query components.
        
        Args:
            extracted_info: Dictionary with extracted fields
            
        Returns:
            QueryComponents dataclass with structured data
        """
        letter_type = extracted_info.get("letter_type", "")
        
        return QueryComponents(
            letter_type=letter_type,
            letter_type_si=self.map_letter_type(letter_type),
            subject=extracted_info.get("subject", ""),
            purpose=extracted_info.get("purpose", ""),
            details=extracted_info.get("details", ""),
            recipient=extracted_info.get("recipient", ""),
            sender=extracted_info.get("sender", ""),
            category=extracted_info.get("letter_category", letter_type),
            register=extracted_info.get("register", "formal"),
        )


# Convenience function for quick query building
def build_sinhala_query(
    letter_type: str = "",
    subject: str = "",
    purpose: str = "",
    details: str = "",
    recipient: str = "",
    sender: str = ""
) -> str:
    """
    Quick utility function to build a Sinhala-aware query.
    
    Args:
        letter_type: Type of letter
        subject: Letter subject
        purpose: Letter purpose
        details: Additional details
        recipient: Letter recipient
        sender: Letter sender
        
    Returns:
        Optimized search query string
    """
    builder = SinhalaQueryBuilder()
    return builder.build_query({
        "letter_type": letter_type,
        "subject": subject,
        "purpose": purpose,
        "details": details,
        "recipient": recipient,
        "sender": sender,
    })


def get_letter_type_sinhala(letter_type: str) -> str:
    """
    Get Sinhala equivalent of an English letter type.
    
    Args:
        letter_type: English letter type
        
    Returns:
        Sinhala letter type string
    """
    builder = SinhalaQueryBuilder()
    return builder.map_letter_type(letter_type)


# For testing
if __name__ == "__main__":
    # Test the query builder
    builder = SinhalaQueryBuilder()
    
    test_info = {
        "letter_type": "request",
        "subject": "නිවාඩු ඉල්ලීම",
        "purpose": "වාර්ෂික නිවාඩු ලබා ගැනීම",
        "details": "පවුලේ වැදගත් කටයුත්තක් සඳහා",
        "recipient": "කළමනාකරු",
        "sender": "සේවක නම",
    }
    
    print("=== Query Builder Test ===")
    print(f"\nInput: {test_info}")
    print(f"\nSingle Query: {builder.build_query(test_info)}")
    print(f"\nMulti Queries: {builder.build_multi_query(test_info)}")
    print(f"\nComponents: {builder.extract_components(test_info).to_dict()}")
    
    # Test letter type mapping
    print("\n=== Letter Type Mapping ===")
    for lt in ["request", "complaint", "application", "apology", "general"]:
        print(f"  {lt} → {builder.map_letter_type(lt)}")
