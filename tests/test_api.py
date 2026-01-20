"""
API test script for the Sinhala Letter RAG system.
Run this after the server is started on localhost:8000
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the root endpoint."""
    print("\n" + "=" * 50)
    print("TEST: Health Check")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_config_endpoint():
    """Test the config endpoint."""
    print("\n" + "=" * 50)
    print("TEST: Config Endpoint")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/config/", timeout=10)
        print(f"Status: {response.status_code}")
        config = response.json()
        print(f"Experiment: {config.get('experiment_name')}")
        print(f"Sinhala Query Builder: {config['retrieval'].get('use_sinhala_query_builder')}")
        print(f"Reranker: {config['retrieval'].get('use_reranker')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_search_endpoint():
    """Test the search endpoint with a Sinhala query."""
    print("\n" + "=" * 50)
    print("TEST: Search Endpoint")
    print("=" * 50)
    
    try:
        # Test with a Sinhala query for request letters
        query = "ඉල්ලීමේ ලිපිය නිවාඩු"
        response = requests.get(
            f"{BASE_URL}/search/",
            params={"query": query, "top_k": 3},
            timeout=30
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Query: {result.get('query')}")
        print(f"Results found: {result.get('result_count')}")
        
        if result.get('results'):
            for i, doc in enumerate(result['results'][:2]):
                print(f"\n  Result {i+1}:")
                meta = doc.get('metadata', {})
                print(f"    Category: {meta.get('letter_category', 'N/A')}")
                print(f"    Type: {meta.get('doc_type', 'N/A')}")
                print(f"    Content preview: {doc.get('content', '')[:100]}...")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_diagnostics():
    """Test the diagnostics endpoint."""
    print("\n" + "=" * 50)
    print("TEST: Diagnostics Endpoint")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/diagnostics/", timeout=30)
        print(f"Status: {response.status_code}")
        diag = response.json()
        print(f"Document count: {diag.get('document_count')}")
        print(f"Embedding model: {diag.get('embedding_model')}")
        print(f"CSV exists: {diag['data_source'].get('csv_exists')}")
        print(f"CSV rows: {diag['data_source'].get('csv_row_count')}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_process_query():
    """Test the process_query endpoint with a Sinhala prompt."""
    print("\n" + "=" * 50)
    print("TEST: Process Query Endpoint")
    print("=" * 50)
    
    try:
        # Test with a Sinhala letter request
        prompt = "මට නිවාඩු ඉල්ලීමක් ලියන්න ඕනෙ. කළමනාකරුට යවන්න"
        
        response = requests.post(
            f"{BASE_URL}/process_query/",
            json={"prompt": prompt},
            timeout=60
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Processing status: {result.get('status')}")
        
        if result.get('extracted_info'):
            print(f"Extracted info:")
            for key, value in result['extracted_info'].items():
                if value:
                    print(f"  - {key}: {value[:50] if isinstance(value, str) else value}...")
        
        if result.get('questions'):
            print(f"Missing info questions:")
            for field, question in result['questions'].items():
                print(f"  - {field}: {question}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False


def run_all_api_tests():
    """Run all API tests."""
    print("\n" + "=" * 60)
    print("🔌 API INTEGRATION TESTS")
    print("=" * 60)
    print(f"Server URL: {BASE_URL}")
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health_check()))
    results.append(("Config", test_config_endpoint()))
    results.append(("Search", test_search_endpoint()))
    results.append(("Diagnostics", test_diagnostics()))
    # Skip process_query if no OpenAI key (requires LLM)
    # results.append(("Process Query", test_process_query()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 API TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All API tests passed!")
    else:
        print("\n⚠️  Some API tests failed.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_api_tests()
    sys.exit(0 if success else 1)
