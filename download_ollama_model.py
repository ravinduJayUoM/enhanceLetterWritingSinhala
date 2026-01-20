"""Download Ollama model via API."""
import requests
import json
import time

model_name = "llama3.2:3b"
print("=" * 60)
print(f"📥 DOWNLOADING MODEL: {model_name}")
print("=" * 60)
print("This will download ~2GB, please wait...")
print()

try:
    response = requests.post(
        "http://localhost:11434/api/pull",
        json={"name": model_name},
        stream=True,
        timeout=600
    )
    
    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                status = data.get("status", "")
                
                # Show progress
                if "total" in data and "completed" in data:
                    total = data["total"]
                    completed = data["completed"]
                    percent = (completed / total) * 100 if total > 0 else 0
                    print(f"\r{status}: {percent:.1f}% ({completed}/{total} bytes)", end="", flush=True)
                else:
                    print(f"\r{status}", end="", flush=True)
        
        print("\n\n✅ Model downloaded successfully!")
        print(f"Model '{model_name}' is ready to use.")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    print("\nAlternative: Open a new PowerShell/CMD window and run:")
    print(f"  ollama pull {model_name}")

print("\n" + "=" * 60)
