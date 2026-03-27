#!/bin/bash
# Sinhala RAG Azure Deployment Script
# Run this on the Azure VM after SSH

set -e  # Exit on error

echo "======================================"
echo "Sinhala Letter RAG - Azure Deployment"
echo "======================================"

# 1. Update system
echo "[1/8] Updating system..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip git curl

# 2. Install Ollama
echo "[2/8] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
sudo systemctl start ollama
sudo systemctl enable ollama
sleep 5

# Verify Ollama is running
if ! systemctl is-active --quiet ollama; then
    echo "ERROR: Ollama failed to start"
    exit 1
fi
echo "✓ Ollama installed and running"

# 3. Pull Aya 8B model
echo "[3/8] Downloading Aya 8B model (4.8GB - this may take 5-10 minutes)..."
ollama pull aya:8b
echo "✓ Aya 8B model downloaded"

# 4. Clone repository
echo "[4/8] Cloning application code..."
cd /home/azureuser
if [ -d "sinhala-letter-rag" ]; then
    echo "Directory exists, pulling latest changes..."
    cd sinhala-letter-rag
    git pull
else
    git clone https://github.com/ravinduJayUoM/enhanceLetterWritingSinhala.git sinhala-letter-rag
    cd sinhala-letter-rag
fi
echo "✓ Code downloaded"

# 5. Setup Python environment
echo "[5/8] Setting up Python environment..."
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Install dependencies
echo "Installing Python packages..."
pip install fastapi uvicorn pandas langchain langchain-community \
    langchain-ollama sentence-transformers faiss-cpu \
    python-multipart pydantic transformers torch

echo "✓ Python environment ready"

# 6. Update configuration
echo "[6/8] Updating configuration..."
cd /home/azureuser/sinhala-letter-rag/rag

# Update config.py to use absolute paths (if needed)
sed -i 's|csv_path = .*|csv_path = "/home/azureuser/sinhala-letter-rag/data/sinhala_letters_v2.csv"|g' config.py

echo "✓ Configuration updated"

# 7. Create systemd service
echo "[7/8] Creating system service..."
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
ExecStart=/home/azureuser/sinhala-letter-rag/venv/bin/python -m uvicorn sinhala_letter_rag:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
sudo systemctl daemon-reload
sudo systemctl enable sinhala-rag
echo "✓ Service configured"

# 8. Start service
echo "[8/8] Starting API service..."
sudo systemctl start sinhala-rag
sleep 5

# Check service status
if systemctl is-active --quiet sinhala-rag; then
    echo "✓ Service started successfully"
else
    echo "ERROR: Service failed to start. Checking logs..."
    sudo journalctl -u sinhala-rag -n 50
    exit 1
fi

# Get public IP
PUBLIC_IP=$(curl -s ifconfig.me)

echo ""
echo "======================================"
echo "✓ DEPLOYMENT COMPLETE!"
echo "======================================"
echo ""
echo "API Base URL: http://$PUBLIC_IP:8000"
echo ""
echo "Test endpoints:"
echo "  Health:     curl http://$PUBLIC_IP:8000/"
echo "  Extract:    curl -X POST http://$PUBLIC_IP:8000/extract/ -H 'Content-Type: application/json' -d '{\"prompt\":\"test\"}'"
echo ""
echo "Useful commands:"
echo "  Check status:  sudo systemctl status sinhala-rag"
echo "  View logs:     sudo journalctl -u sinhala-rag -f"
echo "  Restart:       sudo systemctl restart sinhala-rag"
echo "  Stop:          sudo systemctl stop sinhala-rag"
echo ""
echo "Ollama status:"
echo "  Check:         sudo systemctl status ollama"
echo "  Models:        ollama list"
echo ""
