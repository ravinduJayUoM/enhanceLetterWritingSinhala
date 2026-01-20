# Azure Deployment Plan - Sinhala Letter RAG System

## System Overview

**Components:**
- FastAPI backend (Python 3.13)
- Ollama with Aya 8B model (4.8GB)
- FAISS vector store (local files)
- LaBSE embeddings (HuggingFace)
- 12 document knowledge base

**Requirements:**
- Expose REST APIs on public IP
- Run 24/7 (or on-demand)
- Handle concurrent requests
- No security/authentication for now

---

## Deployment Option: Azure Virtual Machine (Recommended)

### Why Azure VM?
✅ Full control over environment
✅ Easy to install Ollama and dependencies
✅ Cost-effective for development/testing
✅ Can scale vertically if needed
✅ SSH access for management

---

## Step-by-Step Deployment Plan

### **PHASE 1: Azure VM Setup (15 minutes)**

#### 1.1 Create Azure VM
```bash
# Recommended Configuration:
- VM Size: Standard_D4s_v3 (4 vCPUs, 16 GB RAM)
- OS: Ubuntu 22.04 LTS
- Disk: 64 GB Premium SSD (for Aya 8B model + data)
- Region: Choose closest to your users (e.g., Southeast Asia)
- Public IP: Enabled (Static)
- SSH: Enable (port 22)
```

**Why this size?**
- Aya 8B needs ~6-8 GB RAM when loaded
- 4 vCPUs for concurrent request handling
- 16 GB RAM buffer for embeddings + vector search
- Can scale up/down based on usage

**Cost Estimate:**
- Standard_D4s_v3: ~$140-160/month (pay-as-you-go)
- Can use Reserved Instances for 40% discount
- Can stop VM when not in use (pay only for storage)

#### 1.2 Configure Network Security Group (NSG)
```bash
# Inbound Rules:
- Port 22 (SSH): Your IP only
- Port 8000 (FastAPI): 0.0.0.0/0 (public access)
- Port 11434 (Ollama): 127.0.0.1 only (localhost)
```

#### 1.3 Connect to VM
```bash
ssh azureuser@<your-vm-public-ip>
```

---

### **PHASE 2: System Dependencies (20 minutes)**

#### 2.1 Update System
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip git curl
```

#### 2.2 Install Ollama
```bash
# Official Ollama installation
curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version

# Start Ollama service
sudo systemctl start ollama
sudo systemctl enable ollama  # Auto-start on boot

# Check status
sudo systemctl status ollama
```

#### 2.3 Pull Aya 8B Model
```bash
# This downloads 4.8GB - takes 5-10 minutes
ollama pull aya:8b

# Verify model is available
ollama list
```

---

### **PHASE 3: Application Setup (15 minutes)**

#### 3.1 Create Application Directory
```bash
cd /home/azureuser
mkdir -p sinhala-letter-rag
cd sinhala-letter-rag
```

#### 3.2 Transfer Your Code
**Option A: Git (Recommended)**
```bash
# If you push to GitHub first:
git clone https://github.com/ravinduJayUoM/enhanceLetterWritingSinhala.git .
```

**Option B: SCP from Local**
```bash
# From your local Windows machine:
scp -r C:\MSC\code\enhanceLetterWritingSinhala azureuser@<vm-ip>:~/sinhala-letter-rag/
```

**Option C: Manual upload via Azure Portal**
- Zip your project folder
- Upload via Azure Storage Explorer
- Unzip on VM

#### 3.3 Create Python Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### 3.4 Install Python Dependencies
```bash
cd /home/azureuser/sinhala-letter-rag

# Install requirements
pip install -r requirements.txt

# If no requirements.txt, install manually:
pip install fastapi uvicorn pandas langchain langchain-community \
    langchain-ollama sentence-transformers faiss-cpu \
    python-multipart pydantic transformers torch
```

#### 3.5 Create Requirements File (for future)
```bash
# On your local machine first, generate:
pip freeze > requirements.txt

# Then include in your deployment
```

---

### **PHASE 4: Data & Model Files (10 minutes)**

#### 4.1 Transfer Knowledge Base
```bash
# Ensure these directories exist:
mkdir -p /home/azureuser/sinhala-letter-rag/data
mkdir -p /home/azureuser/sinhala-letter-rag/rag/faiss_index

# Transfer from local machine:
scp C:\MSC\code\enhanceLetterWritingSinhala\data\sinhala_letters_v2.csv \
    azureuser@<vm-ip>:~/sinhala-letter-rag/data/

# Transfer FAISS index:
scp -r C:\MSC\code\enhanceLetterWritingSinhala\rag\faiss_index \
    azureuser@<vm-ip>:~/sinhala-letter-rag/rag/
```

#### 4.2 Verify Files
```bash
cd /home/azureuser/sinhala-letter-rag
ls -lh data/
ls -lh rag/faiss_index/
```

---

### **PHASE 5: Configuration Updates (5 minutes)**

#### 5.1 Update Config for Production
```bash
cd /home/azureuser/sinhala-letter-rag/rag
nano config.py
```

**Changes needed:**
```python
# Ensure these settings:
ollama_model = "aya:8b"
ollama_base_url = "http://localhost:11434"  # Local Ollama

# Update paths to absolute:
csv_path = "/home/azureuser/sinhala-letter-rag/data/sinhala_letters_v2.csv"
```

#### 5.2 Test Manual Run
```bash
cd /home/azureuser/sinhala-letter-rag/rag
source ../venv/bin/activate

# Test server manually first:
python -m uvicorn sinhala_letter_rag:app --host 0.0.0.0 --port 8000
```

**Test from your local machine:**
```bash
curl http://<vm-public-ip>:8000/
```

If you see `{"status":"Sinhala Letter RAG System is running"...}` → Success!

Press `Ctrl+C` to stop.

---

### **PHASE 6: Production Service Setup (10 minutes)**

#### 6.1 Create Systemd Service
```bash
sudo nano /etc/systemd/system/sinhala-rag.service
```

**Service file content:**
```ini
[Unit]
Description=Sinhala Letter RAG API Service
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=simple
User=azureuser
WorkingDirectory=/home/azureuser/sinhala-letter-rag/rag
Environment="PATH=/home/azureuser/sinhala-letter-rag/venv/bin"
ExecStart=/home/azureuser/sinhala-letter-rag/venv/bin/python -m uvicorn sinhala_letter_rag:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 6.2 Enable and Start Service
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable sinhala-rag

# Start the service
sudo systemctl start sinhala-rag

# Check status
sudo systemctl status sinhala-rag

# View logs
sudo journalctl -u sinhala-rag -f
```

---

### **PHASE 7: Testing & Verification (10 minutes)**

#### 7.1 API Health Check
```bash
# From VM:
curl http://localhost:8000/

# From your local machine:
curl http://<vm-public-ip>:8000/
```

Expected response:
```json
{
  "status": "Sinhala Letter RAG System is running",
  "rag_processor_available": true,
  "knowledge_base_available": true
}
```

#### 7.2 Test Extraction Endpoint
```bash
curl -X POST "http://<vm-public-ip>:8000/extract/" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "මම අසනිප් නිසා අද නිවාඩුවක් අවශ්‍යයි"
  }'
```

#### 7.3 Test Letter Generation
```bash
curl -X POST "http://<vm-public-ip>:8000/process_query/" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "මම අසනිප් නිසා අද නිවාඩුවක් අවශ්‍යයි"
  }'
```

#### 7.4 Performance Test
```bash
# Install Apache Bench for load testing:
sudo apt install apache2-utils

# Test 100 requests with 10 concurrent:
ab -n 100 -c 10 -H "Content-Type: application/json" \
   -p test_payload.json \
   http://<vm-public-ip>:8000/extract/
```

---

### **PHASE 8: Monitoring & Maintenance**

#### 8.1 Setup Basic Monitoring
```bash
# Install htop for resource monitoring
sudo apt install htop

# Monitor in real-time:
htop
```

#### 8.2 Log Management
```bash
# View application logs:
sudo journalctl -u sinhala-rag -f

# View Ollama logs:
sudo journalctl -u ollama -f

# Rotate logs (prevent disk filling):
sudo nano /etc/logrotate.d/sinhala-rag
```

Add:
```
/var/log/sinhala-rag/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

#### 8.3 Automated Backups
```bash
# Create backup script:
nano ~/backup.sh
```

Content:
```bash
#!/bin/bash
DATE=$(date +%Y%m%d)
BACKUP_DIR=/home/azureuser/backups
mkdir -p $BACKUP_DIR

# Backup data and index
tar -czf $BACKUP_DIR/sinhala-rag-$DATE.tar.gz \
    /home/azureuser/sinhala-letter-rag/data \
    /home/azureuser/sinhala-letter-rag/rag/faiss_index

# Keep only last 7 days
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

Make executable:
```bash
chmod +x ~/backup.sh

# Schedule daily backup (cron):
crontab -e

# Add line:
0 2 * * * /home/azureuser/backup.sh
```

---

## Quick Deployment Commands (Copy-Paste Script)

### On Azure VM (After Creation):
```bash
#!/bin/bash
# Sinhala RAG Deployment Script

# 1. Update system
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip git curl

# 2. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl start ollama
sudo systemctl enable ollama

# 3. Pull Aya 8B
ollama pull aya:8b

# 4. Setup application
cd ~
git clone https://github.com/ravinduJayUoM/enhanceLetterWritingSinhala.git sinhala-letter-rag
cd sinhala-letter-rag

# 5. Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn pandas langchain langchain-community \
    langchain-ollama sentence-transformers faiss-cpu \
    python-multipart pydantic transformers torch

# 6. Create systemd service
sudo tee /etc/systemd/system/sinhala-rag.service > /dev/null <<EOF
[Unit]
Description=Sinhala Letter RAG API Service
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=simple
User=azureuser
WorkingDirectory=/home/azureuser/sinhala-letter-rag/rag
Environment="PATH=/home/azureuser/sinhala-letter-rag/venv/bin"
ExecStart=/home/azureuser/sinhala-letter-rag/venv/bin/python -m uvicorn sinhala_letter_rag:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 7. Start service
sudo systemctl daemon-reload
sudo systemctl enable sinhala-rag
sudo systemctl start sinhala-rag

# 8. Check status
sudo systemctl status sinhala-rag

echo "Deployment complete! Test with: curl http://localhost:8000/"
```

---

## Alternative: Docker Deployment (Advanced)

If you prefer containerization:

### Dockerfile
```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Setup application
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

# Pull Aya model
RUN ollama serve & sleep 10 && ollama pull aya:8b

# Expose port
EXPOSE 8000

# Startup script
CMD ["sh", "-c", "ollama serve & sleep 5 && python -m uvicorn rag.sinhala_letter_rag:app --host 0.0.0.0 --port 8000"]
```

Deploy to Azure Container Instances:
```bash
az container create \
  --resource-group sinhala-rag-rg \
  --name sinhala-rag-container \
  --image <your-docker-image> \
  --cpu 4 --memory 16 \
  --ports 8000 \
  --ip-address Public
```

---

## Cost Optimization Tips

1. **VM Sizing:**
   - Start with Standard_D4s_v3 ($140/month)
   - Monitor usage, downsize to D2s_v3 if possible ($70/month)
   - Use B-series (Burstable) for dev/testing ($30-50/month)

2. **Auto-Shutdown:**
   - Configure VM to auto-shutdown at night if not needed 24/7
   - Save ~50% on costs

3. **Reserved Instances:**
   - Commit to 1-year reserved instance → 40% discount
   - 3-year → 60% discount

4. **Azure for Students:**
   - If you have student subscription, free $100/year credit
   - May cover development costs

---

## Security Considerations (For Future)

Once deployed and tested, add:

1. **API Authentication:**
   - Add FastAPI API key middleware
   - Use Azure API Management

2. **HTTPS:**
   - Install Nginx as reverse proxy
   - Setup Let's Encrypt SSL certificate

3. **Firewall:**
   - Restrict SSH to your IP only
   - Use Azure Bastion for secure access

4. **Monitoring:**
   - Azure Monitor + Application Insights
   - Track API usage, errors, latency

---

## Troubleshooting Common Issues

### Issue 1: Ollama Not Starting
```bash
# Check logs:
sudo journalctl -u ollama -n 50

# Restart:
sudo systemctl restart ollama

# Verify running:
curl http://localhost:11434/api/tags
```

### Issue 2: FastAPI Service Fails
```bash
# Check logs:
sudo journalctl -u sinhala-rag -n 100

# Common causes:
# - Python dependencies missing
# - File paths incorrect
# - Port 8000 already in use

# Test manually:
cd /home/azureuser/sinhala-letter-rag/rag
source ../venv/bin/activate
python -m uvicorn sinhala_letter_rag:app --host 0.0.0.0 --port 8000
```

### Issue 3: Out of Memory
```bash
# Check memory usage:
free -h

# Check which process:
top

# If Ollama using too much:
# - Use smaller model (llama3.2:3b instead of aya:8b)
# - Increase VM size
```

### Issue 4: Slow Response Times
```bash
# Possible causes:
# 1. Model loading on first request (30 seconds)
# 2. Embeddings generation (5-10 seconds)
# 3. CPU bottleneck (need more vCPUs)

# Monitor:
htop

# Optimize:
# - Keep model warm (periodic health checks)
# - Use GPU VM if budget allows (NC-series)
# - Enable response caching for common queries
```

---

## Post-Deployment Checklist

- [ ] VM created and accessible via SSH
- [ ] Ollama installed and running
- [ ] Aya 8B model pulled and available
- [ ] Python environment setup with all dependencies
- [ ] Code transferred and paths configured
- [ ] FAISS index and data files transferred
- [ ] systemd service created and running
- [ ] Port 8000 accessible from public internet
- [ ] Health check endpoint responding
- [ ] Extraction endpoint tested
- [ ] Letter generation endpoint tested
- [ ] Logs configured and monitored
- [ ] Backup script scheduled
- [ ] Documentation updated with public IP

---

## API Endpoints (After Deployment)

Your API will be available at: `http://<vm-public-ip>:8000`

**Endpoints:**
- `GET /` - Health check
- `POST /extract/` - Extract information from prompt
- `POST /process_query/` - Full RAG pipeline
- `POST /generate_letter/` - Generate letter from enhanced prompt
- `GET /search/` - Search knowledge base
- `GET /config/` - Get current configuration

**Example Usage:**
```bash
# Replace <VM-IP> with your actual VM public IP
export API_URL="http://<VM-IP>:8000"

# Health check
curl $API_URL/

# Extract information
curl -X POST "$API_URL/extract/" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "මම අසනිප් නිසා නිවාඩුවක් ඕන"}'
```

---

## Estimated Timeline

| Phase | Task | Time |
|-------|------|------|
| 1 | Azure VM Setup | 15 min |
| 2 | System Dependencies | 20 min |
| 3 | Application Setup | 15 min |
| 4 | Data Transfer | 10 min |
| 5 | Configuration | 5 min |
| 6 | Service Setup | 10 min |
| 7 | Testing | 10 min |
| **Total** | **First Deployment** | **~90 min** |

Subsequent deployments (code updates only): **~5 minutes**

---

## Next Steps

1. **Create Azure VM** with recommended specifications
2. **Run deployment script** from this plan
3. **Test all endpoints** with sample requests
4. **Share public IP** for testing from other devices
5. **Monitor for 24 hours** to ensure stability
6. **Document any issues** encountered
7. **Optimize based on usage** patterns

Would you like me to create the automated deployment script or help with any specific step?
