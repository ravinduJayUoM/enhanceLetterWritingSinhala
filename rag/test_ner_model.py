"""
Test script for the Sinhala Letter NER model.
This demonstrates how the custom NER model extracts structured information from Sinhala text.
"""

import sys
import os
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.models.sinhala_ner import create_model

def test_ner_extraction():
    """Test the custom NER model with example Sinhala letter requests."""
    # Create the NER model
    print("Initializing Sinhala Letter NER model...")
    ner_model = create_model(
        model_name="xlm-roberta-base",
        use_spacy=True,
        use_rules=True
    )
    
    # Example Sinhala letter requests
    test_examples = [
        # Application letter
        """මට ඔබේ ආයතනයේ ගණකාධිකාරී තනතුර සඳහා අයදුම්පතක් ඉදිරිපත් කිරීමට අවශ්‍යයි. මම වයස අවුරුදු 32ක් වන අතර මට වසර 5ක පළපුරුද්දක් ඇත. මගේ විද්යුත් තැපෑල් ලිපිනය example@email.com වේ. මගේ දුරකථන අංකය 0771234567 වේ. මට B.Sc උපාධියක් ඇත.""",
        
        # Complaint letter
        """මම මෙම පැමිණිල්ල ඉදිරිපත් කරන්නේ ජල බිල්පත සම්බන්ධයෙන්. පසුගිය මාසයේ මට ලැබුණු ජල බිල්පත සාමාන්‍යයට වඩා ඉතා ඉහළය. 2023 අප්‍රේල් මාසයේ සිට මෙම ගැටළුව ඇත. කරුණාකර මෙය පරීක්ෂා කර නිවැරදි කරන්න.""",
        
        # Request letter
        """ප්‍රාදේශීය ලේකම් තුමා වෙත, මෙම ලිපිය යොමු කරන්නේ ප්‍රදේශයේ මාර්ගය අලුත්වැඩියා කිරීම සම්බන්ධයෙන් ඉල්ලීමක් කිරීමටයි. මාර්ගය දැඩි ලෙස හානි වී ඇති අතර එය ප්‍රදේශවාසීන්ට බරපතල අපහසුතා ඇති කරයි. ප්‍රදේශවාසීන් වෙනුවෙන් ඉදිරිපත් කරන්නේ, කුමාර ජයතිලක"""
    ]
    
    for i, example in enumerate(test_examples):
        print(f"\n\n=== Test Example {i+1} ===")
        print(f"Input text: {example[:100]}...")
        
        # Extract information using our custom model
        extracted_info = ner_model.extract_info(example)
        
        # Print extracted fields in a readable format
        print("\nExtracted Information:")
        print(json.dumps(extracted_info, indent=2, ensure_ascii=False))
        
        # Compare with rule-based extraction only
        print("\nRule-based extraction only:")
        rule_results = ner_model._rule_based_extraction(example)
        print(json.dumps(rule_results, indent=2, ensure_ascii=False))
        
        print("=" * 80)

if __name__ == "__main__":
    test_ner_extraction()