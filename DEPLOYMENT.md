# Deployment Guide

This guide covers three deployment scenarios: **local development**, **Docker / Docker Compose**, and **cloud (Railway / Render / AWS EC2)**.

---

## Table of Contents

1. [Local Development](#1-local-development)
2. [Docker Compose (recommended for staging)](#2-docker-compose)
3. [Cloud Deployment — Railway](#3-cloud-deployment--railway)
4. [Cloud Deployment — Render](#4-cloud-deployment--render)
5. [Cloud Deployment — AWS EC2](#5-cloud-deployment--aws-ec2)
6. [Environment Variables Reference](#6-environment-variables-reference)
7. [Database Setup](#7-database-setup)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Local Development

### Requirements
- Python 3.10+
- PostgreSQL 14+
- Git

### Steps

```bash
# 1. Clone
git clone https://github.com/durgamp/goldenbatch24March26.git
cd goldenbatch24March26

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Environment file (copy template, fill in your values)
cp .env.example .env               # or create .env manually (see Section 6)

# 5. Create database
psql -U postgres -c "CREATE DATABASE goldenbatch;"

# 6a. Streamlit only
streamlit run app.py

# 6b. Full stack (Streamlit + FastAPI + DB)
python run.py
```

URLs:
- Dashboard → http://localhost:8501
- API docs  → http://localhost:8000/docs

---

## 2. Docker Compose

Docker Compose runs PostgreSQL, FastAPI, and Streamlit as separate containers.

### 2.1 Create a `Dockerfile` for the app

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501
CMD ["python", "run.py"]
```

### 2.2 Create `docker-compose.yml`

```yaml
version: "3.9"

services:
  db:
    image: postgres:16
    restart: always
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: changeme
      POSTGRES_DB: goldenbatch
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  app:
    build: .
    restart: always
    depends_on:
      - db
    env_file:
      - .env
    environment:
      DATABASE_URL: postgresql://postgres:changeme@db:5432/goldenbatch
    ports:
      - "8000:8000"
      - "8501:8501"

volumes:
  pgdata:
```

### 2.3 Build and run

```bash
docker compose up --build
```

To run in the background:

```bash
docker compose up -d --build
```

To stop:

```bash
docker compose down
```

To destroy data volumes as well:

```bash
docker compose down -v
```

---

## 3. Cloud Deployment — Railway

[Railway](https://railway.app) can host both the app and a managed PostgreSQL database for free (starter plan).

### Steps

1. Push code to GitHub (already done).
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo** → select `goldenbatch24March26`.
3. Railway detects Python automatically. Set the **start command**:
   ```
   python run.py
   ```
4. Add a **PostgreSQL** plugin from the Railway dashboard. Railway injects `DATABASE_URL` automatically.
5. Add remaining environment variables under **Variables**:
   ```
   API_HOST=0.0.0.0
   API_PORT=8000
   STREAMLIT_PORT=8501
   ALLOWED_ORIGINS=https://<your-railway-domain>
   ALLOWED_HOSTS=<your-railway-domain>
   ```
6. Railway builds and deploys. Access both services via the generated domain.

> **Note:** Railway exposes one port per service. If you need both Streamlit (8501) and FastAPI (8000), deploy them as two separate Railway services from the same repo with different start commands.

---

## 4. Cloud Deployment — Render

### Streamlit (Web Service)

1. New **Web Service** → connect GitHub repo.
2. Build command: `pip install -r requirements.txt`
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Add environment variables (see Section 6).

### FastAPI (Web Service)

1. New **Web Service** → same repo.
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables.

### PostgreSQL

1. New **PostgreSQL** database on Render.
2. Copy the **Internal Database URL** and set it as `DATABASE_URL` in both web services.

---

## 5. Cloud Deployment — AWS EC2

### 5.1 Launch an EC2 instance

- AMI: Ubuntu 22.04 LTS
- Instance type: t3.medium (minimum recommended)
- Security group: open ports **22** (SSH), **8000** (API), **8501** (Streamlit)

### 5.2 Install dependencies on the server

```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip git postgresql

# Clone the repo
git clone https://github.com/durgamp/goldenbatch24March26.git
cd goldenbatch24March26

# Virtual env
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create .env
nano .env   # paste and save your environment variables
```

### 5.3 Set up PostgreSQL

```bash
sudo -u postgres psql -c "CREATE USER gbuser WITH PASSWORD 'strongpassword';"
sudo -u postgres psql -c "CREATE DATABASE goldenbatch OWNER gbuser;"
```

Update `DATABASE_URL` in `.env`:
```
DATABASE_URL=postgresql://gbuser:strongpassword@localhost:5432/goldenbatch
```

### 5.4 Run with systemd (keeps app alive on reboot)

Create `/etc/systemd/system/goldenbatch.service`:

```ini
[Unit]
Description=Golden Batch Analytics
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/goldenbatch24March26
EnvironmentFile=/home/ubuntu/goldenbatch24March26/.env
ExecStart=/home/ubuntu/goldenbatch24March26/.venv/bin/python run.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable goldenbatch
sudo systemctl start goldenbatch
sudo systemctl status goldenbatch
```

### 5.5 (Optional) Nginx reverse proxy

```nginx
# /etc/nginx/sites-available/goldenbatch
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8000;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/goldenbatch /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

---

## 6. Environment Variables Reference

| Variable | Example | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://user:pass@host:5432/goldenbatch` | PostgreSQL connection string |
| `API_HOST` | `127.0.0.1` | FastAPI bind address (`0.0.0.0` for cloud) |
| `API_PORT` | `8000` | FastAPI port |
| `STREAMLIT_PORT` | `8501` | Streamlit port |
| `ALLOWED_ORIGINS` | `http://localhost:8501` | CORS allowed origins (comma-separated) |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1` | Trusted hosts (comma-separated) |

Create a `.env` file in the project root. **Never commit this file.**
A `.env.example` template (no real values) is safe to commit.

---

## 7. Database Setup

The ORM models in `db/models.py` define the schema. To create tables:

```python
# Run once (from project root with .venv active)
python - <<'EOF'
from db.connection import engine
from db import models
models.Base.metadata.create_all(bind=engine)
print("Tables created.")
EOF
```

Or add this to your app startup sequence in `run.py`.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | venv not active | `source .venv/bin/activate` |
| `could not connect to server` | PostgreSQL not running | `sudo systemctl start postgresql` |
| Streamlit shows "data not loaded" | No CSV uploaded yet | Upload a long-format CSV on the Overview page |
| CORS error in browser | `ALLOWED_ORIGINS` mismatch | Add your frontend URL to `ALLOWED_ORIGINS` in `.env` |
| Port already in use | Another process on 8000/8501 | `lsof -i :8000` then `kill <PID>` |
| Locust can't connect | FastAPI not running | Start with `python run.py` before running Locust |
