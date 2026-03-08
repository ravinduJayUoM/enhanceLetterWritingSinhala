# Oracle Cloud Free Tier Deployment Guide
# Sinhala Letter RAG System

## Why Oracle Cloud Free Tier?
- ✅ **24GB RAM** (more than enough for Aya 8B)
- ✅ **Forever free** (no credit card expiry)
- ✅ 2 OCPUs (ARM-based, sufficient performance)
- ✅ 200GB storage
- ✅ 10TB/month bandwidth
- ✅ No time limit

---

## Step 1: Create Oracle Cloud Account (10 minutes)

1. Go to: https://www.oracle.com/cloud/free/
2. Click **"Start for free"** button
3. Fill in details:
   - **Country/Territory:** Select your country
   - **Email:** Your email address
   - **First Name / Last Name:** Your name
   - Click **"Verify my email"**

4. Check email and click verification link

5. Complete registration:
   - **Cloud Account Name:** Choose unique name (e.g., `sinhalarag123`)
   - **Home Region:** Choose closest:
     - **Mumbai** (ap-mumbai-1) - Best for Sri Lanka
     - Or Singapore (ap-singapore-1)
   - **Password:** Create strong password

6. **Payment verification:**
   - Need credit/debit card for verification (NO CHARGE)
   - Oracle verifies identity only
   - Will NOT charge unless you manually upgrade

7. Wait 2-5 minutes for account provisioning

8. You'll be redirected to Oracle Cloud Console

---

## Step 2: Create Always Free VM Instance (15 minutes)

### 2.1 Navigate to Compute
1. In Oracle Cloud Console, click **☰ hamburger menu** (top left)
2. Navigate to: **Compute** → **Instances**
3. Click **"Create Instance"** button

### 2.2 Configure Instance

**Name and Placement:**
- **Name:** `sinhala-rag-vm`
- **Create in compartment:** (root) - default is fine
- **Placement:** Keep defaults
- **Availability domain:** Select any available

**Image and Shape:**
1. **Image:** 
   - Click **"Change Image"**
   - Select **"Canonical Ubuntu"**
   - Choose **"Ubuntu 22.04"** (or latest 22.04 version)
   - Click **"Select Image"**

2. **Shape:**
   - Click **"Change Shape"**
   - Select **"Ampere"** processor type
   - Choose **"VM.Standard.A1.Flex"**
   - **IMPORTANT:** Configure to Always Free limits:
     - **Number of OCPUs:** 2 (or up to 4 if available)
     - **Amount of memory (GB):** 12 (or up to 24 if available)
   - You should see **"Always Free-eligible"** label
   - Click **"Select Shape"**

**Networking:**
- **Primary VNIC information:**
  - **Virtual cloud network:** Select existing or create new
    - If new, click **"Create new virtual cloud network"**
    - Name: `sinhala-rag-vcn`
    - Keep defaults
  - **Subnet:** Select public subnet
  - **Assign a public IPv4 address:** ✅ **CHECK THIS BOX**
  - Leave other defaults

**Add SSH Keys:**
- **SSH keys:** 
  - Select **"Generate a key pair for me"**
  - Click **"Save Private Key"** → Save as `sinhala-rag-vm.key`
  - Click **"Save Public Key"** (optional backup)
  - **IMPORTANT:** Keep this key file safe!

**Boot Volume:**
- **Boot volume:** Keep defaults (47-50 GB)
- **Use in-transit encryption:** Leave unchecked

### 2.3 Create Instance
1. Review configuration
2. Click **"Create"** button at bottom
3. Wait 2-5 minutes for instance to provision
4. Status will change: **Provisioning** → **Running** (orange → green)

---

## Step 3: Configure Network Security (5 minutes)

### 3.1 Create Ingress Rule for Port 8000
1. Still on Instance Details page, scroll down to **"Primary VNIC"** section
2. Click on **Subnet name** (e.g., `subnet-xxx`)
3. In Subnet details, click **Security Lists** (on left)
4. Click on the **Default Security List** name
5. Click **"Add Ingress Rules"** button

**Configure Ingress Rule:**
- **Stateless:** Leave unchecked
- **Source Type:** CIDR
- **Source CIDR:** `0.0.0.0/0` (allow from anywhere)
- **IP Protocol:** TCP
- **Source Port Range:** Leave blank
- **Destination Port Range:** `8000`
- **Description:** `FastAPI application port`
- Click **"Add Ingress Rules"**

### 3.2 Verify Existing Rules
You should see these rules:
- ✅ SSH (port 22) - already exists
- ✅ Port 8000 - just added

---

## Step 4: Configure OS-Level Firewall (Important!)

Oracle Linux uses firewall rules at OS level too. We need to allow port 8000 through iptables.

**We'll do this after SSH connection (next step)**

---

## Step 5: Connect to Instance via SSH (5 minutes)

### 5.1 Get Instance IP Address
1. Go back to Instance Details page
2. Note the **Public IP Address** (e.g., `144.24.xxx.xxx`)

### 5.2 Fix SSH Key Permissions

**On Windows (PowerShell):**
```powershell
# Move key to .ssh directory
Move-Item "C:\Users\YourName\Downloads\sinhala-rag-vm.key" "C:\Users\YourName\.ssh\"

# Fix permissions (important for Windows)
icacls "C:\Users\YourName\.ssh\sinhala-rag-vm.key" /inheritance:r
icacls "C:\Users\YourName\.ssh\sinhala-rag-vm.key" /grant:r "$($env:USERNAME):R"
```

### 5.3 SSH to Instance

**Option A: PowerShell (Windows)**
```powershell
ssh -i C:\Users\YourName\.ssh\sinhala-rag-vm.key ubuntu@144.24.xxx.xxx
```

**Option B: Oracle Cloud Shell (Web-based)**
1. Click **Cloud Shell** icon (>_) at top right of Oracle Console
2. Upload your SSH key:
   ```bash
   # Upload key via Cloud Shell menu: Upload → Select your .key file
   chmod 600 sinhala-rag-vm.key
   ssh -i sinhala-rag-vm.key ubuntu@144.24.xxx.xxx
   ```

First connection will ask:
```
Are you sure you want to continue connecting (yes/no)?
```
Type: `yes` and press Enter

You should see Ubuntu welcome message!

---

## Step 6: Deploy Application (25 minutes)

### 6.1 Update System and Configure Firewall
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3.11 python3.11-venv python3-pip git curl

# Configure OS firewall to allow port 8000
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT
sudo netfilter-persistent save
```

### 6.2 Install Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
sudo systemctl start ollama
sudo systemctl enable ollama

# Verify Ollama is running
sudo systemctl status ollama
# Should show "active (running)"
```

### 6.3 Download Aya 8B Model (10-15 minutes)
```bash
# Pull Aya 8B model (4.8GB - takes time)
ollama pull aya:8b

# Verify model is available
ollama list
# Should show: aya:8b
```

### 6.4 Clone Repository
```bash
# Clone your application
cd ~
git clone https://github.com/ravinduJayUoM/enhanceLetterWritingSinhala.git sinhala-letter-rag

cd sinhala-letter-rag
```

### 6.5 Setup Python Environment
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install fastapi uvicorn pandas langchain langchain-community \
    langchain-ollama sentence-transformers faiss-cpu \
    python-multipart pydantic transformers torch
```

### 6.6 Create Systemd Service
```bash
sudo tee /etc/systemd/system/sinhala-rag.service > /dev/null <<'EOF'
[Unit]
Description=Sinhala Letter RAG API Service
After=network.target ollama.service
Requires=ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/sinhala-letter-rag/rag
Environment="PATH=/home/ubuntu/sinhala-letter-rag/venv/bin"
ExecStart=/home/ubuntu/sinhala-letter-rag/venv/bin/python -m uvicorn sinhala_letter_rag:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

### 6.7 Start Service
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start on boot
sudo systemctl enable sinhala-rag

# Start service
sudo systemctl start sinhala-rag

# Check status
sudo systemctl status sinhala-rag
# Should show "active (running)" in green
```

### 6.8 View Logs (Optional)
```bash
# Watch logs in real-time
sudo journalctl -u sinhala-rag -f

# Press Ctrl+C to exit
```

---

## Step 7: Test Your Deployment (5 minutes)

### 7.1 Test from Instance (Local)
```bash
# Health check
curl http://localhost:8000/

# Should return:
# {"status":"Sinhala Letter RAG System is running"...}
```

### 7.2 Test from External (Your Computer)

**Get your public IP:**
```bash
curl ifconfig.me
# Or check Instance Details page in Oracle Console
```

**Test from your Windows PowerShell:**
```powershell
# Replace with your actual IP
$VM_IP = "144.24.xxx.xxx"

# Health check
Invoke-RestMethod "http://$VM_IP:8000/"

# Test extraction
$body = @{
    prompt = "මම අසනිප් නිසා නිවාඩුවක් ඕන"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
    -Uri "http://$VM_IP:8000/extract/" `
    -Body $body `
    -ContentType "application/json"
```

**Or test in browser:**
- Open: `http://YOUR_VM_IP:8000/`
- You should see the health check JSON

---

## Step 8: Performance Monitoring

### 8.1 Check Resource Usage
```bash
# Install htop for monitoring
sudo apt install htop

# Monitor resources
htop
# Press F10 or Q to exit
```

### 8.2 Check Memory Usage
```bash
# Free memory
free -h

# Should show:
# - Total: 24GB
# - Used: ~8-10GB (with Aya 8B loaded)
# - Available: ~14GB
```

### 8.3 Monitor Service Logs
```bash
# Last 100 lines
sudo journalctl -u sinhala-rag -n 100

# Live tail
sudo journalctl -u sinhala-rag -f
```

---

## Management Commands

### Service Management
```bash
# Check status
sudo systemctl status sinhala-rag

# Restart service
sudo systemctl restart sinhala-rag

# Stop service
sudo systemctl stop sinhala-rag

# Start service
sudo systemctl start sinhala-rag

# View logs
sudo journalctl -u sinhala-rag -f
```

### Ollama Management
```bash
# Check Ollama status
sudo systemctl status ollama

# List models
ollama list

# Pull new model
ollama pull model-name

# Remove model (free space)
ollama rm llama3.2:3b
```

### Update Application Code
```bash
cd ~/sinhala-letter-rag
git pull origin main
sudo systemctl restart sinhala-rag
```

---

## Oracle Cloud Console Management

### Stop/Start Instance (Save Resources)
1. Go to: Compute → Instances
2. Click on your instance
3. Click **"Stop"** button (instance will stop, saves CPU)
4. Click **"Start"** to restart
5. **Note:** IP address stays the same (static)

### Monitor Usage
1. Go to: Governance → Tenancy Usage
2. View resource consumption
3. Verify "Always Free" resources usage

### View Instance Metrics
1. Instance Details page
2. Click **"Metrics"** (left menu)
3. View CPU, Memory, Network usage graphs

---

## Troubleshooting

### Issue 1: Cannot connect to port 8000
**Solution:**
```bash
# Check if service is running
sudo systemctl status sinhala-rag

# Check if port is open (OS firewall)
sudo iptables -L INPUT -n | grep 8000

# If not listed, add rule:
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT
sudo netfilter-persistent save

# Restart service
sudo systemctl restart sinhala-rag
```

### Issue 2: Service fails to start
**Solution:**
```bash
# View detailed logs
sudo journalctl -u sinhala-rag -n 100

# Common causes:
# 1. Ollama not running
sudo systemctl status ollama
sudo systemctl restart ollama

# 2. Python dependencies missing
source ~/sinhala-letter-rag/venv/bin/activate
pip install -r ~/sinhala-letter-rag/requirements.txt

# 3. Wrong paths in service file
sudo nano /etc/systemd/system/sinhala-rag.service
# Verify paths are correct for ubuntu user

# Restart after fixes
sudo systemctl daemon-reload
sudo systemctl restart sinhala-rag
```

### Issue 3: Out of memory
**Solution:**
```bash
# Check memory
free -h

# If Aya 8B using too much, switch to smaller model:
ollama pull llama3.2:3b
# Update config.py: ollama_model = "llama3.2:3b"
# Restart service

# Or remove unused models:
ollama rm model-name
```

### Issue 4: Slow performance
**Note:** ARM processors are different from x86
- First request loads model (30-60 seconds)
- Subsequent requests faster (5-10 seconds)
- This is normal for free tier
- Still perfectly usable for development/thesis

---

## Cost Verification

### Ensure You're Using Free Tier:
1. Oracle Console → Governance → **Tenancy Usage**
2. Check usage under **"Always Free Resources"**
3. Verify:
   - ✅ Compute: VM.Standard.A1.Flex (2-4 OCPUs, 12-24GB)
   - ✅ Block Storage: 50-200 GB
   - ✅ Public IP: 1-2 reserved
4. **Should show $0.00 cost**

**Important:** As long as you stay within Always Free limits, you will NEVER be charged.

---

## API Endpoints

Your API is now available at: `http://YOUR_ORACLE_VM_IP:8000`

**Available Endpoints:**
- `GET /` - Health check
- `POST /extract/` - Extract information from Sinhala prompt
- `POST /process_query/` - Full RAG pipeline (extract + retrieve + enhance)
- `POST /generate_letter/` - Generate letter from enhanced prompt
- `GET /search/?query=xxx&top_k=5` - Search knowledge base
- `GET /config/` - View current configuration

**Example Usage:**
```bash
# Replace with your actual IP
export API_URL="http://144.24.xxx.xxx:8000"

# Health check
curl $API_URL/

# Extract information
curl -X POST "$API_URL/extract/" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "මම අසනිප් නිසා නිවාඩුවක් ඕන"}'

# Full pipeline
curl -X POST "$API_URL/process_query/" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "මම ගුණසේකර විද්‍යාලයේ ගුරු වැඩට අයදුම් කරන්නම්"}'
```

---

## Backup and Disaster Recovery

### Create Manual Backup
```bash
# Backup application data
cd ~
tar -czf backup-$(date +%Y%m%d).tar.gz sinhala-letter-rag/data sinhala-letter-rag/rag/faiss_index

# Download to local machine (from your computer):
scp -i sinhala-rag-vm.key ubuntu@YOUR_IP:~/backup-*.tar.gz C:\Backups\
```

### Oracle Boot Volume Backup
1. Instance page → Resources → **Boot Volume**
2. Click on boot volume name
3. Click **"Create Manual Backup"**
4. Keeps snapshot of entire VM

---

## Summary - Your Deployment Checklist

- [ ] Oracle Cloud account created
- [ ] VM.Standard.A1.Flex instance created (Always Free)
- [ ] Ubuntu 22.04 installed
- [ ] Public IP assigned
- [ ] Security list: Port 8000 ingress rule added
- [ ] OS firewall: Port 8000 allowed
- [ ] SSH connection working
- [ ] Ollama installed and running
- [ ] Aya 8B model downloaded
- [ ] Python environment setup
- [ ] Application cloned from GitHub
- [ ] Systemd service created and running
- [ ] API accessible from external IP
- [ ] Health check returns success
- [ ] Extract endpoint tested
- [ ] Full pipeline tested

---

## Next Steps

1. **Test thoroughly** - Try all endpoints with different Sinhala prompts
2. **Document API URL** - Share with stakeholders/testers
3. **Monitor for 24 hours** - Ensure stability
4. **Setup automated backups** - Cron job for daily backups
5. **Consider adding:**
   - Nginx reverse proxy (optional)
   - SSL certificate (Let's Encrypt)
   - API authentication (FastAPI middleware)
   - Monitoring (Prometheus + Grafana)

---

## Support Resources

- **Oracle Cloud Docs:** https://docs.oracle.com/en-us/iaas/
- **Always Free Tier Details:** https://www.oracle.com/cloud/free/
- **Community Forums:** https://community.oracle.com/

---

**Congratulations!** 🎉 

Your Sinhala Letter RAG system is now deployed on Oracle Cloud's Always Free Tier with:
- ✅ 24GB RAM
- ✅ Aya 8B model
- ✅ Public API access
- ✅ $0 cost forever
- ✅ Perfect for your MSc thesis!
