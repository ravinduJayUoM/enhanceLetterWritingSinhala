# Using Local Models (FREE Alternative to Azure OpenAI)

Your system now supports **free local models** via Ollama - no API costs!

---

## Option 1: Ollama (Recommended - Easiest)

### 1. Install Ollama
Download from: https://ollama.com/download
- Windows: Download and run the installer
- It will install and start automatically

### 2. Download a Model
Open PowerShell and run:
```powershell
# Small, fast model (3B parameters, ~2GB)
ollama pull llama3.2:3b

# OR larger, better model (8B parameters, ~4.7GB)
ollama pull llama3:8b

# OR Mistral (7B, good multilingual support)
ollama pull mistral
```

### 3. Verify Ollama is Running
```powershell
ollama list
```
You should see your downloaded models.

### 4. Your Code is Already Configured!
The default in `config.py` is now set to use Ollama:
```python
provider: LLMProvider = LLMProvider.OLLAMA
ollama_model: str = "llama3.2:3b"
```

### 5. Start Your Server
```powershell
cd rag
python -m uvicorn sinhala_letter_rag:app --host 0.0.0.0 --port 8000
```

You should see:
```
Using Ollama (local): llama3.2:3b
```

---

## Option 2: HuggingFace Models (Advanced)

If you want more control, use HuggingFace models directly:

### 1. Install Required Packages
```powershell
pip install transformers torch accelerate
```

### 2. Update Config
In `config.py`, change:
```python
provider: LLMProvider = LLMProvider.HUGGINGFACE
huggingface_model: str = "meta-llama/Llama-3.2-3B-Instruct"
```

### 3. Models will download automatically on first use

---

## Recommended Models for Sinhala Support

| Model | Size | Sinhala Support | Best For |
|-------|------|-----------------|----------|
| `llama3.2:3b` | 2GB | Good | Fast, lightweight |
| `llama3:8b` | 4.7GB | Better | Balanced performance |
| `mistral` | 4.1GB | Good | Multilingual |
| `gemma2:9b` | 5.4GB | Good | High quality |

---

## Cost Comparison

| Option | Cost | Setup Time |
|--------|------|------------|
| **Ollama (local)** | ✅ FREE | 5 minutes |
| **HuggingFace (local)** | ✅ FREE | 10 minutes |
| Azure OpenAI API | 💰 ~$0.03/1K tokens | Instant (if working) |

---

## Testing Your Setup

1. Start Ollama (it should auto-start after install)
2. Verify model is downloaded: `ollama list`
3. Start your server
4. Test with your previous API call:
   ```powershell
   $body = '{"prompt":"Write a formal leave request letter"}'
   Invoke-WebRequest -Uri "http://localhost:8000/process_query/" -Method POST -Body $body -ContentType "application/json"
   ```

---

## Troubleshooting

**"Ollama not found"**
- Make sure Ollama is running (check system tray)
- Restart Ollama: `ollama serve`

**"Model not found"**
- Download the model: `ollama pull llama3.2:3b`

**Server says "Using Azure OpenAI"**
- Check `config.py` - make sure `provider = LLMProvider.OLLAMA`

---

## Next Steps

1. Install Ollama
2. Download `llama3.2:3b` model
3. Restart your server
4. Test your API - it should work without any API keys!

Let me know when Ollama is installed and I'll help you test it!
