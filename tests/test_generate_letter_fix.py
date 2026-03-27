"""
Test script to verify the /generate-letter endpoint fix.
This simulates what happens when the endpoint is called.
"""

import requests
import time

# Configure the server URL
SERVER_URL = "http://localhost:8000"  # Change this to your Oracle Cloud IP if testing remote

def test_generate_letter():
    """Test the /generate_letter endpoint."""
    print("Testing /generate_letter endpoint...")
    print("=" * 60)
    
    # Sample enhanced prompt (this would come from /process_query)
    test_prompt = """You are writing an official Sinhala letter.

Details extracted:
- Type: request
- Recipient: විදුහල්පති
- Subject: නිවාඩු අවසරය ඉල්ලීම

Retrieved examples and templates will guide your writing.

Please write a formal letter requesting leave from school."""
    
    payload = {
        "enhanced_prompt": test_prompt
    }
    
    print(f"Sending request to {SERVER_URL}/generate_letter/")
    print(f"Prompt length: {len(test_prompt)} characters")
    print()
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{SERVER_URL}/generate_letter/",
            json=payload,
            timeout=130  # Slightly longer than server timeout
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ SUCCESS! (took {elapsed:.1f}s)")
            print()
            print("Generated letter:")
            print("-" * 60)
            print(result.get("generated_letter", "No letter returned"))
            print("-" * 60)
            return True
        else:
            print(f"✗ FAILED! Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.Timeout:
        elapsed = time.time() - start_time
        print(f"✗ TIMEOUT! Request timed out after {elapsed:.1f}s")
        print("This suggests the server is hanging.")
        return False
    except requests.ConnectionError:
        print(f"✗ CONNECTION ERROR! Cannot connect to {SERVER_URL}")
        print("Make sure the server is running.")
        return False
    except Exception as e:
        print(f"✗ ERROR! {str(e)}")
        return False

def test_other_endpoints():
    """Test other endpoints to ensure they still work."""
    print("\nTesting other endpoints for comparison...")
    print("=" * 60)
    
    # Test /extract
    print("\n1. Testing /extract endpoint...")
    try:
        response = requests.post(
            f"{SERVER_URL}/extract/",
            json={"prompt": "මම විදුහලට නිවාඩු අවසරයක් ඉල්ලීමට අවශ්‍යයි"},
            timeout=60
        )
        if response.status_code == 200:
            print("   ✓ /extract works")
        else:
            print(f"   ✗ /extract failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ /extract error: {str(e)}")
    
    # Test /process_query
    print("\n2. Testing /process_query endpoint...")
    try:
        response = requests.post(
            f"{SERVER_URL}/process_query/",
            json={"prompt": "මම විදුහලට නිවාඩු අවසරයක් ඉල්ලීමට අවශ්‍යයි"},
            timeout=60
        )
        if response.status_code == 200:
            print("   ✓ /process_query works")
        else:
            print(f"   ✗ /process_query failed: {response.status_code}")
    except Exception as e:
        print(f"   ✗ /process_query error: {str(e)}")

if __name__ == "__main__":
    print("Sinhala Letter RAG - /generate_letter Fix Test")
    print("=" * 60)
    print()
    
    # Test the fixed endpoint
    success = test_generate_letter()
    
    # Test other endpoints for comparison
    test_other_endpoints()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Test completed successfully!")
    else:
        print("✗ Test failed. Check the server logs for details.")
    print("=" * 60)
