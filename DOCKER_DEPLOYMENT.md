# QUASAR-SUBNET Docker Deployment Guide

## Docker Images

### 1. Challenge Container
**Image Name**: `quasar-subnet/challenge:latest`

**Build:**
```bash
cd c:\quasar-kimi\QUASAR-SUBNET
docker build -t quasar-subnet/challenge:latest -f challenge/Dockerfile .
```

**Run Locally:**
```bash
docker run -d \
  --name quasar-challenge \
  -p 8080:8080 \
  -v $(pwd)/data:/data \
  -e DATASET_PATH=/data/docmath.jsonl \
  -e SAMPLES_PER_EVALUATION=10 \
  quasar-subnet/challenge:latest
```

### 2. Miner API
**Image Name**: `quasar-subnet/miner:latest`

**Build:**
```bash
docker build -t quasar-subnet/miner:latest -f miner/Dockerfile .
```

**Run Locally:**
```bash
docker run -d \
  --name quasar-miner \
  -p 8000:8000 \
  -e MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct \
  quasar-subnet/miner:latest
```

### 3. Validator Node
**Image Name**: `quasar-subnet/validator:latest`

**Build:**
```bash
docker build -t quasar-subnet/validator:latest -f validator/Dockerfile .
```

**Run Locally:**
```bash
docker run -d \
  --name quasar-validator \
  -e NETUID=24 \
  -e CHALLENGE_URL=http://challenge:8080 \
  quasar-subnet/validator:latest
```

---

## Hosting Options

### Option 1: Docker Hub (Recommended)

**Push to Docker Hub:**
```bash
# Login to Docker Hub
docker login

# Tag images
docker tag quasar-subnet/challenge:latest yourusername/quasar-challenge:latest
docker tag quasar-subnet/miner:latest yourusername/quasar-miner:latest
docker tag quasar-subnet/validator:latest yourusername/quasar-validator:latest

# Push images
docker push yourusername/quasar-challenge:latest
docker push yourusername/quasar-miner:latest
docker push yourusername/quasar-validator:latest
```

**Pull and Run:**
```bash
docker pull yourusername/quasar-challenge:latest
docker run -d -p 8080:8080 yourusername/quasar-challenge:latest
```

### Option 2: GitHub Container Registry (GHCR)

**Push to GHCR:**
```bash
# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Tag images
docker tag quasar-subnet/challenge:latest ghcr.io/yourusername/quasar-challenge:latest
docker tag quasar-subnet/miner:latest ghcr.io/yourusername/quasar-miner:latest
docker tag quasar-subnet/validator:latest ghcr.io/yourusername/quasar-validator:latest

# Push images
docker push ghcr.io/yourusername/quasar-challenge:latest
docker push ghcr.io/yourusername/quasar-miner:latest
docker push ghcr.io/yourusername/quasar-validator:latest
```

### Option 3: Render (Cloud Hosting)

**Challenge Container on Render:**
- Create a new Web Service
- Select "Docker" as runtime
- Image: `yourusername/quasar-challenge:latest`
- Environment Variables:
  - `DATASET_PATH=/data/docmath.jsonl`
  - `SAMPLES_PER_EVALUATION=10`
  - `TIMEOUT_SECS=300`
- Port: 8080

**Miner API on Render:**
- Create a new Web Service
- Select "Docker" as runtime
- Image: `yourusername/quasar-miner:latest`
- Environment Variables:
  - `MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct`
- Port: 8000

**Validator Node on Render:**
- Create a new Web Service
- Select "Docker" as runtime
- Image: `yourusername/quasar-validator:latest`
- Environment Variables:
  - `NETUID=24`
  - `CHALLENGE_URL=https://your-challenge.onrender.com`
  - `NETWORK=finney`

### Option 4: AWS ECS

**Create ECR Repository:**
```bash
aws ecr create-repository --repository-name quasar-challenge
aws ecr create-repository --repository-name quasar-miner
aws ecr create-repository --repository-name quasar-validator
```

**Push to ECR:**
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag quasar-subnet/challenge:latest YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/quasar-challenge:latest
docker push YOUR_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/quasar-challenge:latest
```

### Option 5: Google Cloud Run

**Push to GCR:**
```bash
# Tag for GCR
docker tag quasar-subnet/challenge:latest gcr.io/PROJECT_ID/quasar-challenge:latest

# Push to GCR
docker push gcr.io/PROJECT_ID/quasar-challenge:latest

# Deploy to Cloud Run
gcloud run deploy quasar-challenge \
  --image gcr.io/PROJECT_ID/quasar-challenge:latest \
  --platform managed \
  --region us-central1 \
  --port 8080
```

---

## Recommended Deployment Strategy

### For Development:
- Use `docker-compose.yml` to run all services locally
- Test the full flow before deploying

### For Production:
1. **Challenge Container**: Host on Render (free tier available)
   - URL: `https://quasar-challenge.onrender.com`
   
2. **Miner APIs**: Each miner hosts their own API
   - Can be on Render, AWS, GCP, or self-hosted
   
3. **Validator Nodes**: Run on dedicated servers (AWS/GCP)
   - Need persistent connection to Bittensor
   - Run 24/7 for weight submission

---

## Quick Start with Docker Compose

**Run all services locally:**
```bash
cd c:\quasar-kimi\QUASAR-SUBNET
docker-compose up --build
```

**Access endpoints:**
- Challenge: `http://localhost:8080`
- Miner: `http://localhost:8000`
- Validator: (runs in background)

---

## Environment Variables

### Challenge Container:
- `DATASET_PATH` - Path to docmath.jsonl
- `CHALLENGE_HOST` - Host to bind to (default: 0.0.0.0)
- `CHALLENGE_PORT` - Port to bind to (default: 8080)
- `SAMPLES_PER_EVALUATION` - Number of samples per evaluation (default: 50)
- `TIMEOUT_SECS` - Total evaluation timeout (default: 300)

### Miner API:
- `MODEL_PATH` - HuggingFace model path (default: Qwen/Qwen2.5-0.5B-Instruct)
- `HOST` - Host to bind to (default: 0.0.0.0)
- `PORT` - Port to bind to (default: 8000)
- `MAX_NEW_TOKENS` - Max tokens to generate (default: 100)

### Validator Node:
- `NETUID` - Subnet UID (default: 24)
- `WALLET_NAME` - Bittensor wallet name (default: validator)
- `HOTKEY` - Bittensor hotkey name (default: default)
- `CHALLENGE_URL` - Challenge container URL
- `NETWORK` - Bittensor network (default: finney)
