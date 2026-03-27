"""Quick test to verify Azure OpenAI connection and list available deployments."""
import os
import sys
import requests

# Credentials must be set via environment variables:
#   AZURE_OPENAI_API_KEY
#   AZURE_OPENAI_ENDPOINT
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://sinhala-letter-openai.openai.azure.com/")
api_version = "2023-05-15"

if not api_key:
    print("ERROR: Set the AZURE_OPENAI_API_KEY environment variable before running this test.")
    sys.exit(1)

print("=" * 60)
print("🔍 AZURE OPENAI CONNECTION TEST")
print("=" * 60)
print(f"Endpoint: {endpoint}")
print(f"API Version: {api_version}")
print()

# Test 1: Check resource health
print("📋 Test 1: Checking resource accessibility...")
try:
    # Try to list models endpoint instead
    url = f"{endpoint}openai/models?api-version={api_version}"
    headers = {
        "api-key": api_key
    }
    
    response = requests.get(url, headers=headers)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ Resource is accessible!")
        data = response.json()
        models = data.get("data", [])
        if models:
            print(f"   Available models: {len(models)}")
    elif response.status_code == 401:
        print("❌ Authentication failed - API key might be wrong")
    elif response.status_code == 404:
        print("⚠️  Resource endpoint not found")
    else:
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Connection failed: {e}")

print()

# Test 2: List deployments
print("📋 Test 2: Listing available deployments...")
try:
    url = f"{endpoint}openai/deployments?api-version={api_version}"
    headers = {
        "api-key": api_key
    }
    
    response = requests.get(url, headers=headers)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        deployments = data.get("data", [])
        
        if deployments:
            print(f"✅ Found {len(deployments)} deployment(s):")
            for dep in deployments:
                model = dep.get("model", "unknown")
                dep_id = dep.get("id", "unknown")
                status = dep.get("status", "unknown")
                print(f"  - Name: {dep_id}")
                print(f"    Model: {model}")
                print(f"    Status: {status}")
        else:
            print("⚠️  No deployments found - the resource exists but has no model deployments.")
            print("   You MUST create a deployment to use the API.")
    elif response.status_code == 404:
        print("⚠️  No deployments endpoint available or no deployments created")
    else:
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"❌ Connection failed: {e}")

print()
print("=" * 60)
print("💡 Next Steps:")
print("=" * 60)
print("1. If no deployments found, go to: https://oai.azure.com")
print("2. Create a deployment (e.g., gpt-35-turbo)")
print("3. Use the deployment name in your environment variable")
print("4. Restart your server")
print("=" * 60)
