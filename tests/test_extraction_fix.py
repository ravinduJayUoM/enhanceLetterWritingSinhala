"""
Test the fixed extraction prompt
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag'))

from rag.sinhala_letter_rag import LetterDatabase, RAGProcessor

# Test prompts
TEST_PROMPTS = [
    {
        "name": "Leave Request",
        "prompt": "මම අසනිප් නිසා අද රැකියාවට පැමිණිය නොහැක. කරුණාකර අද සඳහා නිවාඩු අවසරයක් ලබා දෙන්න."
    },
    {
        "name": "Job Application",
        "prompt": "මම ගුණසේකර රාජකීය විද්‍යාලයේ ඉංග්‍රීසි ගුරු තනතුරට අයදුම් කිරීමට කැමතියි. මට බීඑ උපාධියක් සහ වසර 5ක ඉගැන්වීමේ පළපුරුද්ද ඇත."
    },
    {
        "name": "Complaint Letter",
        "prompt": "ජනවාරි 15 වන දින මා විසින් ඔබගේ ආයතනයෙන් මිළදී ගත් රෙෆ්‍රිජරේටරය නිසි ලෙස ක්‍රියා නොකරයි. කරුණාකර මෙය හදා දෙන්න හෝ ආපසු ගෙවන්න."
    }
]

def print_separator(char="=", length=80):
    print(char * length)

def test_extraction():
    print_separator()
    print("TESTING FIXED EXTRACTION PROMPT")
    print_separator()
    print()
    
    # Initialize
    print("Initializing RAG processor...")
    letter_db = LetterDatabase()
    letter_db.build_knowledge_base()
    rag_processor = RAGProcessor(letter_db)
    print("✓ Initialized\n")
    
    # Test each prompt
    for i, test in enumerate(TEST_PROMPTS, 1):
        print_separator("-")
        print(f"TEST {i}: {test['name']}")
        print_separator("-")
        print(f"\nPrompt:\n{test['prompt']}\n")
        
        print("Extracting information...")
        try:
            extracted = rag_processor.extract_key_info(test['prompt'])
            
            if "error" in extracted:
                print(f"❌ ERROR: {extracted['error']}\n")
            else:
                print("✓ Extraction successful!\n")
                print("Extracted Information:")
                for key, value in extracted.items():
                    # Check if we got actual values or placeholder text
                    is_placeholder = (
                        value and isinstance(value, str) and 
                        ("ලිපි" in value or "වර්ගය" in value or "ලබන්නා" in value or "යවන්නා" in value)
                    )
                    
                    status = "❌ PLACEHOLDER" if is_placeholder else "✓"
                    if value and value != "" and value != "N/A":
                        print(f"  {status} {key}: {value}")
                    else:
                        print(f"  - {key}: (empty)")
                
                # Overall assessment
                print()
                has_valid_type = extracted.get("letter_type") in ["application", "request", "complaint", "general", "notification"]
                has_real_values = any(
                    v and v != "" and v != "N/A" and 
                    not any(placeholder in str(v) for placeholder in ["ලිපි", "වර්ගය", "ලබන්නා", "යවන්නා"])
                    for k, v in extracted.items() if k != "letter_type"
                )
                
                if has_valid_type:
                    print("✓ Letter type correctly identified as English value")
                else:
                    print(f"❌ Letter type is invalid: '{extracted.get('letter_type')}'")
                
                if has_real_values:
                    print("✓ Contains extracted values (not just placeholders)")
                else:
                    print("❌ All values are empty or placeholders")
                
        except Exception as e:
            print(f"❌ EXCEPTION: {str(e)}\n")
            import traceback
            traceback.print_exc()
        
        print()
    
    print_separator()
    print("EXTRACTION TEST COMPLETE")
    print_separator()

if __name__ == "__main__":
    test_extraction()
