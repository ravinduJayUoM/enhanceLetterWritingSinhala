"""
Test the /extract/ endpoint for information extraction only
"""
import requests
import json

BASE_URL = "http://localhost:8000"

# Test prompts
TEST_PROMPTS = [
    "මම අසනිප් නිසා අද රැකියාවට පැමිණිය නොහැක. කරුණාකර අද සඳහා නිවාඩු අවසරයක් ලබා දෙන්න.",
    "මම ගුණසේකර රාජකීය විද්‍යාලයේ ඉංග්‍රීසි ගුරු තනතුරට අයදුම් කිරීමට කැමතියි. මට බීඑ උපාධියක් සහ වසර 5ක ඉගැන්වීමේ පළපුරුද්ද ඇත.",
    "ජනවාරි 15 වන දින මා විසින් ඔබගේ ආයතනයෙන් මිළදී ගත් රෙෆ්‍රිජරේටරය නිසි ලෙස ක්‍රියා නොකරයි. කරුණාකර මෙය හදා දෙන්න හෝ ආපසු ගෙවන්න.",
]

def test_extract_endpoint():
    print("="*80)
    print("Testing /extract/ Endpoint")
    print("="*80)
    print()
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}")
        print(f"{'='*80}")
        print(f"\nPrompt:\n{prompt}\n")
        
        # Call the extract endpoint
        try:
            response = requests.post(
                f"{BASE_URL}/extract/",
                json={"prompt": prompt}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print("Extracted Information:")
                extracted = result.get("extracted_info", {})
                
                for key, value in extracted.items():
                    if value and value != "":
                        status = "✓" if value != "error" else "✗"
                        print(f"  {status} {key}: {value}")
                    else:
                        print(f"  - {key}: (empty)")
                
                # Check quality
                print()
                has_error = "error" in extracted
                has_valid_type = extracted.get("letter_type") in ["application", "request", "complaint", "general"]
                has_content = any(v and v != "" for k, v in extracted.items() if k != "letter_type")
                
                if has_error:
                    print("❌ Extraction failed with error")
                elif not has_valid_type:
                    print(f"⚠️  Invalid letter_type: {extracted.get('letter_type')}")
                elif not has_content:
                    print("⚠️  No information extracted")
                else:
                    print("✓ Extraction successful")
                    
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"   {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection Error: Make sure the server is running on localhost:8000")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "="*80)
    print("Test Complete")
    print("="*80)

if __name__ == "__main__":
    print("\nMake sure the server is running: python -m uvicorn sinhala_letter_rag:app\n")
    test_extract_endpoint()
