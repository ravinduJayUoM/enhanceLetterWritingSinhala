"""
Test extraction with Aya 8B model
"""
import requests
import json

# Test prompts
test_prompts = [
    "මම අසනිප් නිසා අද රැකියාවට පැමිණිය නොහැක. කරුණාකර අද සඳහා නිවාඩු අවසරයක් ලබා දෙන්න.",
    "මම ගුණසේකර රාජකීය විද්‍යාලයේ ඉංග්‍රීසි ගුරු තනතුරට අයදුම් කිරීමට කැමතියි. මට බීඑ උපාධියක් සහ වසර 5ක ඉගැන්වීමේ පළපුරුද්ද ඇත.",
    "ජනවාරි 15 වන දින මා විසින් ඔබගේ ආයතනයෙන් මිළදී ගත් රෙෆ්‍රිජරේටරය නිසි ලෙස ක්‍රියා නොකරයි. කරුණාකර මෙය හදා දෙන්න හෝ ආපසු ගෙවන්න."
]

print("="*80)
print("TESTING AYA 8B MODEL - EXTRACTION")
print("="*80)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n--- Test {i} ---")
    print(f"Prompt: {prompt}\n")
    
    response = requests.post(
        "http://127.0.0.1:8000/extract/",
        json={"prompt": prompt}
    )
    
    if response.status_code == 200:
        data = response.json()
        print("✓ Extraction successful!\n")
        print("Extracted Information:")
        for key, value in data.items():
            if value and value != "":
                print(f"  {key}: {value}")
        
        # Check quality
        has_valid_type = data.get("letter_type") in ["application", "request", "complaint", "general"]
        has_content = any(v and v != "" for k, v in data.items() if k != "letter_type")
        
        print()
        if has_valid_type:
            print("✓ Letter type correctly identified")
        else:
            print(f"❌ Invalid letter type: {data.get('letter_type')}")
        
        if has_content:
            print("✓ Contains extracted content")
        else:
            print("❌ No content extracted")
    else:
        print(f"❌ Request failed: {response.status_code}")
        print(response.text)
    
    print()

print("="*80)
