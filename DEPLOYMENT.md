# 🚀 Production Deployment Guide

## Option 1: AWS EC2 with GPU (Recommended for Production)

### Step 1: Launch EC2 Instance
```bash
# Recommended: g4dn.xlarge (NVIDIA T4 GPU, ~$0.50/hour)
# Or: g5.xlarge (NVIDIA A10G, ~$1/hour) for faster inference
```

### Step 2: Setup Server
```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Clone your repo
git clone https://github.com/YOUR_USERNAME/summarization-platform.git
cd summarization-platform

# Run with GPU
docker-compose -f docker-compose.gpu.yml up -d
```

### Step 3: Setup Domain & SSL
```bash
# Install Nginx & Certbot
sudo apt install nginx certbot python3-certbot-nginx

# Configure Nginx
sudo nano /etc/nginx/sites-available/summarize
```

---

## Option 2: Google Cloud Run (Serverless, Auto-scaling)

### Step 1: Build & Push Docker Image
```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Build and push
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/summarization-api

# Deploy
gcloud run deploy summarization-api \
  --image gcr.io/YOUR_PROJECT_ID/summarization-api \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4 \
  --timeout 300 \
  --allow-unauthenticated
```

---

## Option 3: Railway.app (Easiest, Free Tier)

### Step 1: One-Click Deploy
1. Go to [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub"
3. Select your repo
4. Railway auto-detects Dockerfile and deploys
5. Get free subdomain: `your-app.up.railway.app`

---

## Option 4: Hugging Face Spaces (Free GPU!)

### Step 1: Create Space
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create new Space → Select "Docker"
3. Choose **GPU** hardware (free T4 available)
4. Upload your code or connect GitHub

### Step 2: Create app.py for Gradio UI
```python
import gradio as gr
from app.services.summarization import HierarchicalSummarizer

summarizer = HierarchicalSummarizer()

async def summarize(text, mode):
    result = await summarizer.summarize(text, mode=mode)
    return result.summary_medium

demo = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(label="Text to Summarize", lines=10),
        gr.Dropdown(["student", "researcher", "business"], label="Mode")
    ],
    outputs=gr.Textbox(label="Summary"),
    title="🧠 AI Summarization Platform"
)

demo.launch()
```

---

## Option 5: Render.com (Simple PaaS)

### Step 1: Connect GitHub
1. Go to [render.com](https://render.com)
2. New → Web Service → Connect GitHub repo
3. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Deploy!

---

## Production Checklist

### Security
- [ ] Enable HTTPS (SSL certificate)
- [ ] Set strong SECRET_KEY in .env
- [ ] Enable rate limiting
- [ ] Add authentication (JWT/OAuth)
- [ ] Sanitize all inputs

### Performance
- [ ] Use GPU instance for fast inference
- [ ] Enable Redis caching
- [ ] Use CDN for frontend (Cloudflare)
- [ ] Set up load balancer for scaling

### Monitoring
- [ ] Setup logging (CloudWatch/Datadog)
- [ ] Add health check endpoints
- [ ] Configure alerts for errors
- [ ] Monitor GPU/memory usage

### Database
- [ ] Use PostgreSQL instead of SQLite
- [ ] Setup automated backups
- [ ] Configure connection pooling

---

## Cost Estimates (Monthly)

| Platform | Specs | Cost |
|----------|-------|------|
| AWS g4dn.xlarge | T4 GPU, 4 vCPU, 16GB | ~$380/mo (on-demand) |
| AWS g4dn.xlarge | Spot Instance | ~$120/mo |
| Google Cloud Run | Serverless, pay-per-use | ~$50-200/mo |
| Railway | 8GB RAM, no GPU | ~$20/mo |
| Hugging Face Spaces | Free T4 GPU | **FREE** |
| Render | 2GB RAM | ~$25/mo |

---

## Quick Start Commands

```bash
# Local Development
python -m uvicorn app.main:app --reload --port 8000

# Docker Production
docker-compose up -d

# Docker with GPU
docker-compose -f docker-compose.gpu.yml up -d
```
