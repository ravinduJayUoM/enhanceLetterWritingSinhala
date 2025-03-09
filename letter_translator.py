import openai
import os
from typing import Dict, Tuple

class LetterTranslator:
    def __init__(self, api_key=None):
        """
        Initialize the letter translator service
        
        Args:
            api_key: OpenAI API key (will use environment variable if not provided)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment variables")
        
        openai.api_key = self.api_key
    
    def translate_and_classify(self, letter_content: str) -> Tuple[str, str, Dict]:
        """
        Translate Sinhala letter to English and determine its type
        
        Args:
            letter_content: The content of the letter in Sinhala
            
        Returns:
            Tuple containing (translated_text, letter_type, metadata)
        """
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional translator and document analyst specializing in Sinhala to English translation."},
                    {"role": "user", "content": f"""
                    Please translate the following letter from Sinhala to English:
                    
                    {letter_content}
                    
                    After translating, please determine the letter type from these categories:
                    - application
                    - request
                    - complaint
                    - appreciation
                    - invitation
                    - notice/ announcement
                    - apology
                    - general
                    
                    Return your response in JSON format with the following structure:
                    {{
                        "translation": "the English translation",
                        "letter_type": "determined letter type",
                        "confidence": "high/medium/low",
                        "keywords_found": ["list", "of", "key", "terms", "that", "helped", "classification"]
                    }}
                    """}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = response.choices[0].message.content
            parsed_result = eval(result)
            
            return (
                parsed_result.get("translation", ""),
                parsed_result.get("letter_type", "general"),
                {
                    "confidence": parsed_result.get("confidence", "low"),
                    "keywords_found": parsed_result.get("keywords_found", [])
                }
            )
            
        except Exception as e:
            print(f"Error translating letter: {e}")
            return "", "general", {"confidence": "low", "keywords_found": []}
