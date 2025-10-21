# 🚀 Deployment Guide

## Docker Deployment

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM
- 10GB free disk space

### Quick Start
```bash
# Build the container
docker-compose build

# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop the service
docker-compose down
```

The API will be available at `http://localhost:8000`

### Environment Variables

Edit `docker-compose.yml` to customize:
```yaml
environment:
  - MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
  - DEVICE=cpu  # Use 'cuda' for NVIDIA GPU
  - MAX_LENGTH=512
```

### Health Check
```bash
curl http://localhost:8000/health
```

### Monitoring
```bash
# View container stats
docker stats edge-llm-iot-monitor

# View logs
docker logs -f edge-llm-iot-monitor
```

---

## Production Deployment

### 1. Cloud Deployment (AWS EC2 / GCP / Azure)
```bash
# On your server
git clone <your-repo>
cd edge-llm-iot-monitor

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Deploy
docker-compose up -d

# Setup nginx reverse proxy
sudo apt install nginx
```

**Nginx config** (`/etc/nginx/sites-available/iot-monitor`):
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 2. Kubernetes Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: edge-llm-iot
spec:
  replicas: 2
  selector:
    matchLabels:
      app: edge-llm-iot
  template:
    metadata:
      labels:
        app: edge-llm-iot
    spec:
      containers:
      - name: api
        image: your-registry/edge-llm-iot:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            memory: "8Gi"
            cpu: "4"
          requests:
            memory: "4Gi"
            cpu: "2"
```

### 3. Edge Device Deployment (Raspberry Pi / Jetson)
```bash
# Use ARM-compatible base image
# Modify Dockerfile:
FROM python:3.9-slim-bullseye

# Use INT8 quantization for better performance
# In .env:
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct-GGUF
DEVICE=cpu
MAX_LENGTH=256
```

---

## Scaling & Optimization

### Horizontal Scaling
```yaml
# docker-compose.yml with load balancer
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api-1
      - api-2

  api-1:
    build: .
    environment:
      - DEVICE=cpu

  api-2:
    build: .
    environment:
      - DEVICE=cpu
```

### Caching Layer

Add Redis for response caching:
```yaml
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

### Monitoring Stack
```yaml
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

---

## Security Best Practices

### 1. API Authentication
```python
# Add to api/main.py
from fastapi.security import HTTPBearer

security = HTTPBearer()

@app.post("/query")
async def query(credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Verify token
    pass
```

### 2. Rate Limiting
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/query")
@limiter.limit("10/minute")
async def query():
    pass
```

### 3. HTTPS with Let's Encrypt
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs

# Check resources
docker stats

# Rebuild
docker-compose build --no-cache
```

### Out of memory

Reduce batch size in `.env`:
```bash
MAX_LENGTH=256
```

### Slow inference

- Use GPU if available (`DEVICE=cuda`)
- Enable model quantization
- Use ONNX runtime
- Implement response caching

---

## Performance Tuning

### For Production
```python
# api/main.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,  # Multiple workers
        limit_concurrency=10,
        timeout_keep_alive=30
    )
```

### For Edge Devices
```bash
# Reduce memory footprint
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
DEVICE=cpu
MAX_LENGTH=256
BATCH_SIZE=1
```

---

## Backup & Recovery

### Backup ChromaDB
```bash
docker-compose exec edge-llm-iot tar -czf /app/chroma_backup.tar.gz /app/chroma_db
docker cp edge-llm-iot:/app/chroma_backup.tar.gz ./backups/
```

### Restore
```bash
docker cp ./backups/chroma_backup.tar.gz edge-llm-iot:/app/
docker-compose exec edge-llm-iot tar -xzf /app/chroma_backup.tar.gz
```