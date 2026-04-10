# SinhalaLipi — Deployment Guide

## Overview

| Component | Technology | Host |
|-----------|-----------|------|
| Backend API | Python FastAPI + Uvicorn | Google Cloud VM (systemd service) |
| Frontend | React (static build) | Same VM, served via Nginx |
| Database | SQLite (users.db) | Same VM filesystem |
| Vector Store | FAISS index | Same VM filesystem |
| LLM | Google Gemini 2.5 Flash | External API |
| Domain | sinhalalipi.lk | domains.lk + Let's Encrypt SSL |

---

## Part 1 — Initial Deployment from Scratch (Google Cloud)

### 1.1 Google Cloud Setup

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project (e.g. `sinhala-letter-rag`)
3. Enable **Compute Engine**
4. Create a VM instance:

| Setting | Value |
|---------|-------|
| Name | `sinhala-rag-vm` |
| Region | `asia-south1` (or closest) |
| Machine type | `e2-standard-4` (4 vCPU, 16GB RAM) |
| Boot disk | Ubuntu 22.04 LTS, 50GB |
| Firewall | Allow HTTP and HTTPS traffic |

5. Add a firewall rule for port 8000:
   - Go to **VPC Network → Firewall → Create Firewall Rule**
   - Name: `allow-port-8000`
   - Direction: Ingress, Action: Allow
   - Source: `0.0.0.0/0`, Protocol: TCP, Port: `8000`

6. Note the **External IP** of the VM (e.g. `34.87.52.138`)

---

### 1.2 SSH into the VM

Click **SSH** button in Compute Engine → VM instances, or use gcloud CLI.

---

### 1.3 System Dependencies

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git nodejs npm nginx cloud-guest-utils
```

---

### 1.4 Clone the Repository

```bash
cd ~
git clone https://github.com/ravinduJayUoM/enhanceLetterWritingSinhala.git
cd enhanceLetterWritingSinhala
```

---

### 1.5 Backend Setup

```bash
cd ~/enhanceLetterWritingSinhala/rag
python3 -m venv venv
source venv/bin/activate

pip install fastapi uvicorn pandas langchain langchain-community langchain-openai \
    sentence-transformers faiss-cpu torch transformers ollama python-multipart \
    pydantic python-dotenv langchain-google-genai python-jose[cryptography] bcrypt
```

Create the `.env` file:

```bash
# Create using a text editor — do NOT use echo on Windows or it will corrupt encoding
nano ~/enhanceLetterWritingSinhala/rag/.env
```

Contents:
```
GEMINI_API_KEY=your_gemini_api_key_here
JWT_SECRET_KEY=your_random_secret_key_here
```

> Get a Gemini API key from [aistudio.google.com](https://aistudio.google.com) → Get API Key → Create API key in existing project.

---

### 1.6 Backend Systemd Service

```bash
sudo tee /etc/systemd/system/sinhala-rag.service > /dev/null << 'EOF'
[Unit]
Description=Sinhala Letter RAG System
After=network.target

[Service]
Type=simple
User=healthsyncdev
WorkingDirectory=/home/healthsyncdev/enhanceLetterWritingSinhala/rag
Environment="PATH=/home/healthsyncdev/enhanceLetterWritingSinhala/rag/venv/bin"
EnvironmentFile=/home/healthsyncdev/enhanceLetterWritingSinhala/rag/.env
ExecStart=/home/healthsyncdev/enhanceLetterWritingSinhala/rag/venv/bin/python /home/healthsyncdev/enhanceLetterWritingSinhala/run_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable sinhala-rag
sudo systemctl start sinhala-rag
sudo systemctl status sinhala-rag
```

---

### 1.7 Frontend Build

```bash
cd ~/enhanceLetterWritingSinhala/ui
npm install
npm run build
```

---

### 1.8 Nginx Configuration

```bash
sudo tee /etc/nginx/sites-available/sinhala-letter << 'EOF'
server {
    listen 80;
    server_name sinhalalipi.lk www.sinhalalipi.lk;

    root /home/healthsyncdev/enhanceLetterWritingSinhala/ui/build;
    index index.html;

    # Serve React frontend
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy backend API calls
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/sinhala-letter /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl enable nginx
sudo systemctl restart nginx
```

---

### 1.9 SSL Certificate (Let's Encrypt)

> DNS must be fully propagated before running this step.

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d sinhalalipi.lk -d www.sinhalalipi.lk
```

- Enter your email when prompted
- Choose **Redirect** (option 2) when asked about HTTP → HTTPS

Certbot auto-renews. Verify the renewal timer:

```bash
sudo systemctl status certbot.timer
```

---

### 1.10 DNS Setup (domains.lk)

In the domains.lk DNS manager, add under **Other Resource Records**:

| Name | TTL | Type | Value |
|------|-----|------|-------|
| `sinhalalipi.lk.` | Default | A | `34.87.52.138` |
| `www.sinhalalipi.lk.` | Default | A | `34.87.52.138` |

> domains.lk processes DNS changes once daily between **10:00 PM – 11:00 PM Sri Lanka time**.

---

### 1.11 Open VM Firewall (iptables)

```bash
sudo iptables -I INPUT 1 -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 1 -p tcp --dport 443 -j ACCEPT
sudo iptables -I INPUT 1 -p tcp --dport 8000 -j ACCEPT
sudo apt install -y iptables-persistent
sudo netfilter-persistent save
```

---

## Part 2 — Rollout Deployment

### 2.1 Frontend-Only Update

Use this when you changed anything inside `ui/src/`.

**On your local machine:**

```bash
git add ui/
git commit -m "feat: your change description"
git push
```

**On the VM:**

```bash
cd ~/enhanceLetterWritingSinhala
git pull
cd ui
npm run build
sudo systemctl restart nginx
```

> No backend restart needed.

---

### 2.2 Backend-Only Update

Use this when you changed anything inside `rag/` (API, pipeline, config, etc.).

**On your local machine:**

```bash
git add rag/
git commit -m "feat: your change description"
git push
```

**On the VM:**

```bash
cd ~/enhanceLetterWritingSinhala
git pull
source rag/venv/bin/activate

# Only if new packages were added
pip install -r rag/requirements.txt   # or manually install new packages

sudo systemctl restart sinhala-rag
sudo systemctl status sinhala-rag
```

> No frontend rebuild needed.

---

### 2.3 Full Update (Frontend + Backend)

Use this when both `ui/` and `rag/` have changed.

**On your local machine:**

```bash
git add -A
git commit -m "feat: your change description"
git push
```

**On the VM:**

```bash
cd ~/enhanceLetterWritingSinhala

# Back up .env in case git pull conflicts
cp rag/.env rag/.env.backup

git pull

# Restore .env if overwritten
cp rag/.env.backup rag/.env

# Backend
source rag/venv/bin/activate
pip install python-jose[cryptography] bcrypt  # add any new packages here
sudo systemctl restart sinhala-rag

# Frontend
cd ui
npm run build
sudo systemctl restart nginx
```

---

### 2.4 Verify After Any Deployment

```bash
# Check backend is running
sudo systemctl status sinhala-rag

# Check Nginx is running
sudo systemctl status nginx

# Check backend health
curl http://localhost:8000/

# Check frontend is being served
curl -I http://localhost:80
```

---

## Useful Commands

| Task | Command |
|------|---------|
| View backend logs | `sudo journalctl -u sinhala-rag -f` |
| View Nginx logs | `sudo tail -f /var/log/nginx/error.log` |
| Restart backend | `sudo systemctl restart sinhala-rag` |
| Restart Nginx | `sudo systemctl restart nginx` |
| Check disk space | `df -h` |
| Renew SSL manually | `sudo certbot renew` |
