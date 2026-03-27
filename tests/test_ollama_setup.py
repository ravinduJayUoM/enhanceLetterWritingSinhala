"""Test Ollama installation and setup."""
import requests
import json

print("=" * 60)
print("🧪 OLLAMA SETUP TEST")
print("=" * 60)

# Test 1: Check if Ollama is running
print("\n📋 Test 1: Checking if Ollama is running...")
try:
    response = requests.get("http://localhost:11434/api/tags")
    if response.status_code == 200:
        print("✅ Ollama is running!")
        data = response.json()
        models = data.get("models", [])
        
        if models:
            print(f"✅ Found {len(models)} model(s) installed:")
            for model in models:
                name = model.get("name", "unknown")
                size = model.get("size", 0) / (1024**3)  # Convert to GB
                print(f"   - {name} ({size:.2f} GB)")
        else:
            print("⚠️  Ollama is running but no models are installed yet")
            print("   Run: ollama pull llama3.2:3b")
    else:
        print(f"⚠️  Ollama responded with status: {response.status_code}")
        
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to Ollama")
    print("   Make sure Ollama is installed and running")
    print("   Download from: https://ollama.com/download")
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Test a simple generation (if models exist)
print("\n📋 Test 2: Testing text generation...")
try:
    response = requests.get("http://localhost:11434/api/tags")
    if response.status_code == 200:
        data = response.json()
        models = data.get("models", [])
        
        if models:
            # Use the first available model
            model_name = models[0].get("name")
            print(f"Testing with model: {model_name}")
            
            # Simple generation test
            gen_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Say hello in one word:",
                    "stream": False
                },
                timeout=30
            )
            
            if gen_response.status_code == 200:
                result = gen_response.json()
                print(f"✅ Generation works!")
                print(f"   Response: {result.get('response', 'N/A')[:100]}")
            else:
                print(f"⚠️  Generation failed: {gen_response.status_code}")
        else:
            print("⚠️  No models to test - install one first")
            
except Exception as e:
    print(f"⚠️  Could not test generation: {e}")

print("\n" + "=" * 60)
print("📝 SUMMARY")
print("=" * 60)
print("If Ollama is not running:")
print("  1. Install from: https://ollama.com/download")
print("  2. It should start automatically")
print("")
print("If no models are installed:")
print("  1. Run: ollama pull llama3.2:3b")
print("  2. Wait for download (~2GB)")
print("=" * 60)
