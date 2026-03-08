# Oracle Cloud Deployment Guide - Sinhala Letter RAG System

Complete step-by-step guide to deploy the Sinhala Letter Writing RAG system on Oracle Cloud Free Tier.

**Estimated Time:** 60-90 minutes  
**Cost:** $0 (Always Free Tier)

---

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Phase 1: Network Setup (VCN)](#phase-1-network-setup-vcn)
3. [Phase 2: Create Compute Instance](#phase-2-create-compute-instance)
4. [Phase 3: Configure Security](#phase-3-configure-security)
5. [Phase 4: Deploy Application](#phase-4-deploy-application)
6. [Phase 5: Testing](#phase-5-testing)
7. [Phase 6: Setup Systemd Service](#phase-6-setup-systemd-service)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Oracle Cloud Account
- Sign up at https://www.oracle.com/cloud/free/
- Complete email verification
- Add payment method (required but won't be charged for Always Free resources)

### 2. Local Requirements
- SSH client (built into Windows 10+, macOS, Linux)
- Git repository pushed to GitHub
- Your SSH key pair ready

### 3. Resource Requirements
- **VM Shape:** VM.Standard.A1.Flex (Ampere ARM)
- **OCPUs:** 4 cores
- **Memory:** 24 GB RAM
- **Storage:** 50 GB boot volume
- **OS:** Ubuntu 22.04

---

## Phase 1: Network Setup (VCN)

### Step 1.1: Create Virtual Cloud Network

1. Navigate to **☰ Menu → Networking → Virtual Cloud Networks**

2. Click **"Start VCN Wizard"**

3. Select **"Create VCN with Internet Connectivity"**

4. Configure VCN:
   ```
   VCN Name: sinhala-rag-vcn
   Compartment: (root) or your compartment
   VCN IPv4 CIDR Block: 10.0.0.0/16
   Public Subnet CIDR: 10.0.0.0/24
   Private Subnet CIDR: 10.0.1.0/24
   ```

5. Click **"Next"** → **"Create"**

6. Wait for creation (~30 seconds)

### Step 1.2: Verify Internet Gateway

After VCN creation, verify:

1. Click on your VCN name (**sinhala-rag-vcn**)
2. On the left menu, click **"Internet Gateways"**
3. Verify an Internet Gateway exists (created automatically by wizard)
4. If missing, create one:
   - Click **"Create Internet Gateway"**
   - Name: `internet-gateway`
   - Enable: ☑️ Checked
   - Click **"Create"**

### Step 1.3: Configure Route Table

1. In VCN page, click **"Route Tables"** on left menu
2. Look for a route table with Internet Gateway route
3. If none exists, create **"public-route-table"**:
   - Click **"Create Route Table"**
   - Name: `public-route-table`
   - Click **"+ Another Route Rule"**
   - Target Type: **Internet Gateway**
   - Destination CIDR: `0.0.0.0/0`
   - Target: Select your Internet Gateway
   - Click **"Create"**

### Step 1.4: Configure Security List

1. In VCN page, click **"Security Lists"** on left
2. Click on **"Default Security List"**
3. Click **"Add Ingress Rules"**

**Add Rule 1 - SSH Access:**
```
Source Type: CIDR
Source CIDR: 0.0.0.0/0
IP Protocol: TCP
Source Port Range: (leave blank)
Destination Port Range: 22
Description: SSH access
```

**Add Rule 2 - API Access:**
```
Source Type: CIDR
Source CIDR: 0.0.0.0/0
IP Protocol: TCP
Source Port Range: (leave blank)
Destination Port Range: 8000
Description: FastAPI server
```

4. Click **"Add Ingress Rules"** for each

### Step 1.5: Associate Public Subnet with Correct Route Table

1. In VCN page, click **"Subnets"**
2. Click on the **public subnet** (e.g., "public subnet-sinhala-rag-vcn")
3. Click **"Edit"**
4. Change **Route Table** to **public-route-table** (if you created a custom one)
5. Verify **Subnet Access** is set to **Public Subnet**
6. Click **"Save Changes"**

---

## Phase 2: Create Compute Instance

### Step 2.1: Launch Instance Creation

1. Navigate to **☰ Menu → Compute → Instances**

2. Click **"Create Instance"**

### Step 2.2: Configure Instance

**Basic Information:**
```
Name: sinhala-rag-server
Compartment: (root) or your compartment
Availability Domain: AD-1 (or any available)
Fault Domain: Let Oracle choose
```

**Image:**
```
Operating System: Canonical Ubuntu
Version: 22.04
Image Build: Latest (2025.10.31 or newer)
```

**Shape:**
1. Click **"Change Shape"**
2. Select **"Ampere"** processor type
3. Select **VM.Standard.A1.Flex**
4. Configure:
   ```
   OCPU Count: 4
   Memory (GB): 24
   Network Bandwidth (Gbps): Will auto-adjust
   ```
5. Click **"Select Shape"**

**Networking:**
1. Select **"Select existing virtual cloud network"**
2. Choose: **sinhala-rag-vcn**
3. Select **"Select existing subnet"**
4. Choose: Your **public subnet**
5. ☑️ Check **"Automatically assign public IPv4 address"**
6. Leave Private IP as automatic

**SSH Keys:**
1. Generate new key pair OR upload existing public key
2. **IMPORTANT:** Download and save the private key securely
3. Note the key filename (e.g., `ssh-key-2026-01-20.key`)

**Boot Volume:**
```
Boot Volume Size: 50 GB (default)
Use in-transit encryption: Enabled
```

### Step 2.3: Create Instance

1. Review all settings
2. Click **"Create"**
3. Wait for provisioning (2-5 minutes)
4. Status will change from **Provisioning** → **Running**

### Step 2.4: Note the Public IP

1. Once running, find **Public IP Address** in instance details
2. Copy this IP (e.g., `134.185.83.81`)
3. You'll need this for SSH and API access

---

## Phase 3: Configure Security

### Step 3.1: Fix SSH Key Permissions (Windows)

If using Windows, secure your SSH key:

```powershell
# Navigate to key location
cd C:\path\to\your\key\

# Remove all permissions except yours
icacls ssh-key-2026-01-20.key /inheritance:r
icacls ssh-key-2026-01-20.key /grant:r "$($env:USERNAME):(R)"
```

### Step 3.2: Test SSH Connection

```bash
ssh -i C:\path\to\ssh-key-2026-01-20.key ubuntu@YOUR_PUBLIC_IP
```

**Expected Output:**
```
Welcome to Ubuntu 22.04.5 LTS (GNU/Linux 5.15.0-1068-oracle aarch64)
...
ubuntu@instance-20260120-1753:~$
```

**If connection times out:**
- Verify Internet Gateway exists in VCN
- Verify route table has 0.0.0.0/0 → Internet Gateway rule
- Verify security list has port 22 ingress rule
- Verify subnet is configured as **Public Subnet**
- See [Troubleshooting](#troubleshooting) section

---

## Phase 4: Deploy Application

### Step 4.1: Run Complete Deployment Script

Once SSH'd into the VM, copy and paste this entire script:

```bash
#!/bin/bash
# Sinhala Letter RAG System - Complete Deployment Script
# Run on Ubuntu 22.04 (Oracle Cloud VM.Standard.A1.Flex)

set -e  # Exit on any error

echo "============================================"
echo "Sinhala Letter RAG System - Deployment"
echo "============================================"

# Update system
echo "[1/8] Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install dependencies
echo "[2/8] Installing dependencies..."
sudo apt-get install -y git python3.11 python3.11-venv python3-pip curl build-essential

# Install Ollama
echo "[3/8] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
echo "[4/8] Starting Ollama service..."
sudo systemctl enable ollama
sudo systemctl start ollama
sleep 5  # Wait for service to start

# Verify Ollama is running
if ! sudo systemctl is-active --quiet ollama; then
    echo "ERROR: Ollama service failed to start"
    exit 1
fi

# Pull Aya 8B model (4.8GB download, 5-10 minutes)
echo "[5/8] Downloading Aya 8B model (4.8GB, ~10 minutes)..."
ollama pull aya:8b

# Verify model downloaded
if ! ollama list | grep -q "aya:8b"; then
    echo "ERROR: Aya 8B model failed to download"
    exit 1
fi

# Clone repository
echo "[6/8] Cloning repository..."
cd ~
if [ -d "enhanceLetterWritingSinhala" ]; then
    echo "Repository already exists, pulling latest..."
    cd enhanceLetterWritingSinhala
    git pull
else
    git clone https://github.com/ravinduJayUoM/enhanceLetterWritingSinhala.git
    cd enhanceLetterWritingSinhala
fi

# Create virtual environment
echo "[7/8] Setting up Python virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "[8/8] Installing Python packages..."
pip install --upgrade pip
pip install fastapi uvicorn sentence-transformers faiss-cpu torch transformers ollama pandas langchain-text-splitters langchain-community langchain-openai

# Configure VM firewall
echo "Configuring VM firewall for port 8000..."
sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT
sudo mkdir -p /etc/iptables
sudo iptables-save | sudo tee /etc/iptables/rules.v4 > /dev/null

echo "============================================"
echo "Deployment Complete!"
echo "============================================"
echo ""
echo "To start the server:"
echo "  cd ~/enhanceLetterWritingSinhala"
echo "  source venv/bin/activate"
echo "  python run_server.py"
echo ""
echo "The server will run on: http://0.0.0.0:8000"
echo "Access externally at: http://YOUR_PUBLIC_IP:8000"
echo "============================================"
```

### Step 4.2: Monitor Installation

The script will take **15-20 minutes** total:
- System updates: 2-3 min
- Ollama installation: 30 sec
- **Aya 8B download: 5-10 min** ← Longest step
- Repository clone: 10 sec
- Python packages: 3-5 min

You'll see progress bars and status messages for each step.

### Step 4.3: Manual Start (First Time)

After deployment completes:

```bash
cd ~/enhanceLetterWritingSinhala
source venv/bin/activate
python run_server.py
```

**Expected Output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Loading FAISS index from: rag/faiss_index
INFO:     Loaded 12 documents
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## Phase 5: Testing

### Step 5.1: Test from SSH Session

In the SSH terminal (while server is running):

```bash
# Open a new SSH session (don't close the server)
ssh -i C:\path\to\ssh-key-2026-01-20.key ubuntu@YOUR_PUBLIC_IP

# Test health endpoint
curl http://localhost:8000/health

# Expected: {"status":"healthy"}
```

### Step 5.2: Test from Your Laptop

Open PowerShell/Terminal on your local machine:

```powershell
# Test health endpoint
curl http://YOUR_PUBLIC_IP:8000/health

# Test document count
curl http://YOUR_PUBLIC_IP:8000/documents/count

# Test extraction endpoint
curl -X POST http://YOUR_PUBLIC_IP:8000/extract/ `
  -H "Content-Type: application/json" `
  -d '{
    "text": "මගේ නම රවීන්දු. මම විශ්වවිද්‍යාලයට ලිපියක් යැවීමට කැමතියි."
  }'
```

### Step 5.3: Test Full RAG Pipeline

```powershell
curl -X POST http://YOUR_PUBLIC_IP:8000/generate/ `
  -H "Content-Type: application/json" `
  -d '{
    "prompt": "විශ්වවිද්‍යාලයට තොරතුරු ඉල්ලීම් ලිපියක් ලියන්න"
  }'
```

**Expected:** A properly formatted Sinhala letter with:
- Sender information
- Date
- Address
- Subject
- Body content
- Closing

---

## Phase 6: Setup Systemd Service

To run the server automatically on boot:

### Step 6.1: Create Service File

```bash
sudo nano /etc/systemd/system/sinhala-rag.service
```

Paste this configuration:

```ini
[Unit]
Description=Sinhala Letter RAG System
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/enhanceLetterWritingSinhala
Environment="PATH=/home/ubuntu/enhanceLetterWritingSinhala/venv/bin"
ExecStart=/home/ubuntu/enhanceLetterWritingSinhala/venv/bin/python run_server.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Save: `Ctrl+X`, `Y`, `Enter`

### Step 6.2: Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable sinhala-rag

# Start service now
sudo systemctl start sinhala-rag

# Check status
sudo systemctl status sinhala-rag
```

### Step 6.3: Manage Service

```bash
# View logs
sudo journalctl -u sinhala-rag -f

# Restart service
sudo systemctl restart sinhala-rag

# Stop service
sudo systemctl stop sinhala-rag

# Check if running
sudo systemctl is-active sinhala-rag
```

---

## Troubleshooting

### Problem 1: SSH Connection Timeout

**Symptoms:**
```
ssh: connect to host X.X.X.X port 22: Connection timed out
```

**Solutions:**

1. **Check Internet Gateway:**
   - Go to VCN → Internet Gateways
   - Verify gateway exists and is enabled
   - If missing, create one

2. **Check Route Table:**
   - Go to VCN → Route Tables → public-route-table
   - Verify rule: `0.0.0.0/0` → Internet Gateway
   - If missing, add the rule

3. **Check Security List:**
   - Go to VCN → Security Lists → Default Security List
   - Verify ingress rule for port 22 from `0.0.0.0/0`
   - Verify **Stateless** is **unchecked**

4. **Check Subnet Configuration:**
   - Go to VCN → Subnets → Your public subnet
   - Verify **Subnet Access** = **Public Subnet**
   - Verify **Route Table** points to table with Internet Gateway

5. **Check Network Security Groups:**
   - Go to instance → Primary VNIC
   - If NSG is attached, add port 22 ingress rule to NSG as well

### Problem 2: VM.Standard.A1.Flex Out of Capacity

**Symptoms:**
```
Out of capacity for shape VM.Standard.A1.Flex in availability domain AD-1
```

**Solutions:**

1. **Try Different Availability Domain:**
   - Change from AD-1 to AD-2 or AD-3
   - Some ADs have more capacity

2. **Try Different Region:**
   - Mumbai (ap-mumbai-1)
   - Frankfurt (eu-frankfurt-1)
   - Tokyo (ap-tokyo-1)

3. **Automated Retry Script:**
   Use Oracle Cloud Shell (>_ icon in console):
   ```bash
   # Get your configuration IDs first
   oci compute instance launch --help
   
   # Then run retry loop
   while true; do
     oci compute instance launch \
       --availability-domain "YOUR-AD" \
       --compartment-id "YOUR-COMPARTMENT-ID" \
       --shape VM.Standard.A1.Flex \
       --shape-config '{"ocpus":4,"memoryInGBs":24}' \
       --image-id "YOUR-UBUNTU-IMAGE-ID" \
       --subnet-id "YOUR-SUBNET-ID" \
       --display-name "sinhala-rag-server" \
       --assign-public-ip true && break
     echo "Retrying in 60 seconds..."
     sleep 60
   done
   ```

4. **Use Paid Alternative Temporarily:**
   - VM.Standard.E4.Flex (4 OCPU, 24GB) ~$150/month
   - Can switch to A1.Flex later when capacity available

### Problem 3: Ollama Service Won't Start

**Symptoms:**
```
Failed to start ollama.service: Unit ollama.service not found
```

**Solutions:**

```bash
# Check if Ollama installed
which ollama

# If not found, reinstall
curl -fsSL https://ollama.com/install.sh | sh

# Start manually
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama

# Check logs
sudo journalctl -u ollama -n 50
```

### Problem 4: Aya Model Download Fails

**Symptoms:**
```
Error: failed to pull model
```

**Solutions:**

```bash
# Check Ollama service is running
sudo systemctl status ollama

# Try pulling again with verbose output
OLLAMA_DEBUG=1 ollama pull aya:8b

# Check disk space
df -h

# If low space, increase boot volume:
# Oracle Console → Instance → Boot Volume → Edit → Increase size
```

### Problem 5: Python Package Installation Fails

**Symptoms:**
```
ERROR: Could not build wheels for torch
```

**Solutions:**

```bash
# Install build dependencies
sudo apt-get install -y build-essential python3-dev

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Try installing torch first separately
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then install remaining packages
pip install fastapi uvicorn sentence-transformers faiss-cpu transformers ollama pandas langchain-text-splitters langchain-community langchain-openai
```

### Problem 6: API Returns 500 Error

**Check Server Logs:**

```bash
# If running manually
# Look at terminal output

# If running as service
sudo journalctl -u sinhala-rag -n 100

# Common issues:
# 1. FAISS index missing - check rag/faiss_index/ exists
# 2. Ollama not responding - restart: sudo systemctl restart ollama
# 3. Model not loaded - verify: ollama list
```

### Problem 7: Extraction Returns Garbage Data

**Symptoms:**
Extraction endpoint returns field descriptions instead of actual values.

**Solutions:**

1. **Verify Aya 8B is loaded:**
   ```bash
   ollama list
   # Should show: aya:8b
   ```

2. **Test Ollama directly:**
   ```bash
   ollama run aya:8b "Extract name from: මගේ නම රවීන්දු"
   ```

3. **Check config.py:**
   ```bash
   cat rag/config.py | grep ollama_model
   # Should show: ollama_model="aya:8b"
   ```

4. **Restart services:**
   ```bash
   sudo systemctl restart ollama
   sudo systemctl restart sinhala-rag
   ```

---

## Performance Optimization

### Enable Swap (Recommended for 24GB RAM)

```bash
# Create 8GB swap file
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Verify
free -h
```

### Monitor Resource Usage

```bash
# CPU and memory
htop

# Disk usage
df -h

# Service resource usage
systemctl status sinhala-rag
```

---

## Maintenance

### Update Application Code

```bash
cd ~/enhanceLetterWritingSinhala
git pull
sudo systemctl restart sinhala-rag
```

### Update System Packages

```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo reboot  # If kernel updated
```

### Backup FAISS Index

```bash
# Backup to local machine
scp -i C:\path\to\key ubuntu@YOUR_IP:~/enhanceLetterWritingSinhala/rag/faiss_index/* ./backup/

# Restore
scp -i C:\path\to\key ./backup/* ubuntu@YOUR_IP:~/enhanceLetterWritingSinhala/rag/faiss_index/
```

### View Logs

```bash
# Application logs
sudo journalctl -u sinhala-rag -f

# Ollama logs
sudo journalctl -u ollama -f

# System logs
sudo tail -f /var/log/syslog
```

---

## Security Best Practices

### 1. Restrict SSH Access (Optional)

Update security list to only allow your IP:

```
Source CIDR: YOUR_IP_ADDRESS/32  (instead of 0.0.0.0/0)
```

### 2. Enable Firewall on VM

```bash
# Install UFW
sudo apt-get install -y ufw

# Allow SSH and API
sudo ufw allow 22/tcp
sudo ufw allow 8000/tcp

# Enable firewall
sudo ufw enable
```

### 3. Regular Updates

```bash
# Setup automatic security updates
sudo apt-get install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

---

## Cost Monitoring

**Always Free Resources Used:**
- 1x VM.Standard.A1.Flex: 4 OCPU, 24GB RAM ✓ (within 4 OCPU limit)
- 50GB Boot Volume ✓ (within 200GB limit)
- Public IP ✓ (2 free IPs available)
- Outbound Data Transfer: 10TB/month ✓

**Monitor Usage:**
1. Go to **☰ → Billing & Cost Management → Cost Analysis**
2. Verify "Always Free" tag on resources
3. Check for any charges (should be $0)

---

## Next Steps

1. **Add More Training Data:**
   - Expand `data/sinhala_letters_v2.csv` with 20-30 more examples
   - Rebuild FAISS index
   - Improves retrieval quality

2. **Monitor Performance:**
   - Track extraction accuracy
   - Monitor response times
   - Collect user feedback

3. **Optional Enhancements:**
   - Setup HTTPS with Let's Encrypt
   - Add authentication (API keys)
   - Implement rate limiting
   - Add logging and monitoring (Prometheus/Grafana)

---

## Support

**Issues?**
- Check [Troubleshooting](#troubleshooting) section
- Review service logs: `sudo journalctl -u sinhala-rag -n 100`
- GitHub Issues: https://github.com/ravinduJayUoM/enhanceLetterWritingSinhala/issues

**Useful Commands:**
```bash
# Check everything
sudo systemctl status ollama
sudo systemctl status sinhala-rag
ollama list
curl http://localhost:8000/health

# Restart everything
sudo systemctl restart ollama
sudo systemctl restart sinhala-rag
```

---

## Deployment Checklist

- [ ] Oracle Cloud account created and verified
- [ ] VCN created with Internet Gateway
- [ ] Public subnet configured with correct route table
- [ ] Security list has ports 22 and 8000 open
- [ ] VM instance created (VM.Standard.A1.Flex, 4 OCPU, 24GB)
- [ ] Public IP assigned to instance
- [ ] SSH connection working
- [ ] Deployment script executed successfully
- [ ] Ollama service running
- [ ] Aya 8B model downloaded
- [ ] Python application running
- [ ] Health endpoint responding
- [ ] Systemd service configured and enabled
- [ ] Testing completed from external IP
- [ ] Backup plan established

---

**Deployment Guide Version:** 1.0  
**Last Updated:** January 20, 2026  
**Author:** GitHub Copilot  
**Repository:** https://github.com/ravinduJayUoM/enhanceLetterWritingSinhala
