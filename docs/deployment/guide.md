# Deployment Guide

Deploy the Geo-AI book assistant on a single machine.
This guide covers macOS (Mac Mini / Apple Silicon) and Linux deployment paths.

For model selection and memory budgets, see [deployment-models.md](../../system-design/deployment-models.md).

## Platform differences

The app, indexer, and reranker use ML models (sentence-transformers, fastembed) that benefit from GPU acceleration.
Docker containers on macOS cannot access the Metal GPU, so the app and indexer run natively on macOS.
On Linux with NVIDIA GPU, everything runs in Docker with GPU passthrough.
The `docker-compose.yml` uses profiles to enforce this: `docker compose up -d` starts only Qdrant and PostgreSQL.
Services that need GPU (app, indexer) are behind the `gpu` or `build` profile and must be requested explicitly.

| Component | macOS | Linux (NVIDIA) |
|-----------|-------|----------------|
| Qdrant + Postgres | Docker | Docker |
| Ollama (LLM) | Native (Metal) | Native (CUDA) |
| App (uvicorn) | Native (Metal/MPS) | Docker (CUDA) |
| Indexer | Native (Metal/MPS) | Docker (CUDA) |
| Book builder | Docker (no GPU needed) | Docker |

## Prerequisites

### macOS note: admin vs non-admin users

Installing Docker Desktop and Ollama requires writing to `/Applications`, which needs an admin account.
If your deployment user is non-admin (recommended for servers), install from the admin account, then switch back to the non-admin user for everything else.

### just (task runner)

```bash
brew install just       # macOS
# Linux: see https://just.systems/man/en/installation.html
```

All deployment commands in this guide use `just` recipes.
Run `just --list` to see available commands.

### Docker

On macOS, install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
Download the `.dmg` from the website and drag Docker.app to `/Applications`, or:

```bash
brew install --cask docker-desktop
```

On Linux, install [Docker Engine](https://docs.docker.com/engine/install/) and the [Compose plugin](https://docs.docker.com/compose/install/linux/).
For GPU support, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

After installation, open the app once to complete setup:

```bash
open /Applications/Docker.app    # macOS
```

Docker Desktop runs in the background after first launch.
Enable "Start Docker Desktop when you sign in" in Docker Desktop > Settings > General so it auto-starts on reboot.

### Ollama

Ollama runs natively (not in Docker) so it can access the GPU directly (Metal on macOS, CUDA on Linux).

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

On macOS, you can also download the `.dmg` from [ollama.com/download/mac](https://ollama.com/download/mac).

After installation, start Ollama before pulling models:

```bash
open /Applications/Ollama.app    # macOS, wait for menu bar icon
ollama serve &                   # Linux
```

Pull the LLM model:

```bash
ollama pull qwen3:8b
```

Verify it works:

```bash
ollama run qwen3:8b "Hello, what is semantic segmentation?"
```

You should get a coherent response.
Press Ctrl+D to exit.

**Ollama KV cache tuning (optional):** On Apple Silicon, setting `OLLAMA_KV_CACHE_TYPE=q8_0` halves the KV cache memory with negligible quality impact. Set it before starting Ollama: `launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0` then restart the Ollama app.

### uv (macOS only)

On macOS, the app and indexer run natively (not in Docker) to access the Metal GPU.
This requires uv, which manages Python and dependencies:

```bash
brew install uv
```

`uv sync` auto-downloads the right Python version (3.12+) if it is not already installed.

## 1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/biplovbhandari/geoai-book-assistant.git
cd geoai-book-assistant
```

## 2. Install dependencies (macOS only)

```bash
uv sync --group dev --group indexer
```

Not needed on Linux - the Docker images include all dependencies.

## 3. Configure the environment

```bash
cp .env.example .env
```

Edit `.env` with deployment settings.

**macOS (.env):**

```bash
LLM_MODEL=qwen3:8b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_TIMEOUT_SECONDS=120
LLM_REASONING_EFFORT=none
EVAL_JUDGE_REASONING_EFFORT=none
DATABASE_URL=postgresql+asyncpg://geoai:geoai@localhost:5432/geoai
ADMIN_TOKEN=<generate with: python3 -c "import secrets; print(secrets.token_urlsafe(32))">
EMBEDDING_DIMENSION=1024
RERANKER_PROVIDER=local_cross_encoder
RETRIEVAL_STRATEGY=hybrid
QDRANT_COLLECTION=geoai_book
RATE_LIMIT_PER_MINUTE=30
MAX_CONCURRENT_LLM=1
GRAFANA_ADMIN_PASSWORD=admin
```

`ADMIN_TOKEN` protects the `/analytics/*` and `/metrics` endpoints.
Requests must include `Authorization: Bearer <token>`.
If left empty, all analytics endpoints return 403.

On macOS, `LLM_BASE_URL` and `DATABASE_URL` use `localhost` because the app runs natively on the host alongside Ollama and Docker services.

**Linux (.env):**

```bash
LLM_MODEL=qwen3:8b
LLM_BASE_URL=http://172.17.0.1:11434/v1
LLM_API_KEY=ollama
LLM_REASONING_EFFORT=none
EVAL_JUDGE_REASONING_EFFORT=none
DATABASE_URL=postgresql+asyncpg://geoai:geoai@postgres:5432/geoai
ADMIN_TOKEN=<generate with: python3 -c "import secrets; print(secrets.token_urlsafe(32))">
EMBEDDING_DIMENSION=1024
RERANKER_PROVIDER=local_cross_encoder
RETRIEVAL_STRATEGY=hybrid
QDRANT_COLLECTION=geoai_book
RATE_LIMIT_PER_MINUTE=30
MAX_CONCURRENT_LLM=1
GRAFANA_ADMIN_PASSWORD=admin
```

On Linux, `LLM_BASE_URL` uses the Docker bridge gateway (`172.17.0.1`) and `DATABASE_URL` uses the Docker service name (`postgres`) because the app runs inside Docker.

## 4. Start services and application

### macOS

```bash
just services       # Docker: Qdrant + PostgreSQL + Prometheus + Grafana
just db-migrate     # Apply database migrations
just serve          # Native: uvicorn in production mode with Metal GPU
```

`just serve` starts the API without hot-reload, bound to all interfaces, with graceful shutdown (`--timeout-graceful-shutdown 30` drains in-flight SSE streams before exit).
`just dev` is available for development (adds hot-reload).
The embedding model and reranker run on the Metal GPU via MPS.

To run as a persistent service that survives reboots:

```bash
mkdir -p logs

# Copy the template and edit paths
cp infra/launchd/com.geoai.assistant.plist ~/Library/LaunchAgents/
# Edit ~/Library/LaunchAgents/com.geoai.assistant.plist:
#   Replace REPO_PATH with your clone location (e.g. /Users/yourname/geoai-book-assistant)
#   Replace HOMEBREW_PREFIX with /opt/homebrew (Apple Silicon) or /usr/local (Intel)

launchctl load ~/Library/LaunchAgents/com.geoai.assistant.plist
```

This auto-starts the app on login, restarts on crash, and logs to `logs/app/stdout.log` (raw uvicorn output) and `logs/app/app.log` (structured JSON via RotatingFileHandler).
To stop: `launchctl unload ~/Library/LaunchAgents/com.geoai.assistant.plist`.

### Linux

```bash
just up-prod        # Docker: full stack from GHCR images (app + Qdrant + PostgreSQL)
```

Database migrations run automatically on startup via the entrypoint script.
The app and its ML models use CUDA GPU inside the container.

## 5. Render the book and index content

### macOS

```bash
just render-book    # Docker: Quarto builder (no GPU needed)
just index          # Native: indexer with Metal GPU for embeddings
```

### Linux

```bash
just render-book-prod    # Docker: GHCR Quarto builder image
just index-prod          # Docker: GHCR indexer image with CUDA GPU
```

To rebuild the index from scratch, add `--recreate-collection`:

```bash
just index --recreate-collection          # macOS
just index-prod --recreate-collection     # Linux
```

## 6. Verify

```bash
# Health check
curl -s localhost:8000/health | python3 -m json.tool

# Search (retrieval only)
curl -s -X POST localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"question": "What is U-Net?"}' | python3 -m json.tool

# Chat (streaming, requires Ollama)
curl -N -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"question": "What is semantic segmentation?"}'
```

Check that:
- `/health` shows `retrieval: ready`, `generation: ready`, `chat: ready`, `database: ready`
- `/search` returns chunks with scores
- `/chat` streams tokens via SSE

The book with chat widget is at http://localhost:8000/.
The analytics dashboard is at http://localhost:8000/dashboard/ (enter `ADMIN_TOKEN` to connect).
The Qdrant dashboard is at http://localhost:6333/dashboard.
The Grafana dashboard is at http://localhost:3000 (or `GRAFANA_PORT`, admin/admin or `GRAFANA_ADMIN_PASSWORD`).
The Prometheus UI is at http://localhost:9090 (check Targets page to verify scrape health).

## 7. Public access with Cloudflare Tunnel

Cloudflare Tunnel exposes the app to the internet with HTTPS, without opening ports on your router.

### Install cloudflared

```bash
# macOS
brew install cloudflare/cloudflare/cloudflared

# Linux
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  -o /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared
```

### Authenticate

```bash
cloudflared tunnel login
```

This opens a browser to authorize your Cloudflare account.

### Create a tunnel

```bash
cloudflared tunnel create geoai
```

Note the tunnel ID from the output (a UUID like `abc123de-...`).

### Configure the tunnel

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: <TUNNEL_ID>
credentials-file: ~/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: geoai.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

Replace `<TUNNEL_ID>` with your tunnel ID and `geoai.yourdomain.com` with your domain.

### Create DNS record

```bash
cloudflared tunnel route dns geoai geoai.yourdomain.com
```

### Start the tunnel

```bash
cloudflared tunnel run geoai
```

The app is now reachable at `https://geoai.yourdomain.com`.

### Run as a service (persistent)

To keep the tunnel running after logout:

**macOS (non-admin user):**

```bash
# Copy the template and edit paths
cp infra/launchd/com.geoai.tunnel.plist ~/Library/LaunchAgents/
# Edit ~/Library/LaunchAgents/com.geoai.tunnel.plist:
#   Replace REPO_PATH with your clone location (e.g. /Users/yourname/geoai-book-assistant)
#   Replace HOMEBREW_PREFIX with /opt/homebrew (Apple Silicon) or /usr/local (Intel)

launchctl load ~/Library/LaunchAgents/com.geoai.tunnel.plist
```

To stop: `launchctl unload ~/Library/LaunchAgents/com.geoai.tunnel.plist`.

**macOS (admin user):**

```bash
sudo cloudflared service install
sudo launchctl start com.cloudflare.cloudflared
```

**Linux (systemd):**

```bash
sudo cloudflared service install
sudo systemctl enable --now cloudflared
```

## 8. Updating

When new code merges to main, GHCR images are rebuilt automatically.

### macOS

```bash
git pull
uv sync --group dev --group indexer
just render-book
just index

# Restart the app service
launchctl unload ~/Library/LaunchAgents/com.geoai.assistant.plist
launchctl load ~/Library/LaunchAgents/com.geoai.assistant.plist

# Restart the tunnel (only if tunnel config changed)
launchctl unload ~/Library/LaunchAgents/com.geoai.tunnel.plist
launchctl load ~/Library/LaunchAgents/com.geoai.tunnel.plist
```

If the update includes database schema changes, run `just db-migrate` before restarting the app.

### Linux

```bash
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml pull
just up-prod
just render-book-prod
just index-prod
```

Database migrations run automatically on restart via the entrypoint script.

## 9. Monitoring

### Health check

```bash
curl -s localhost:8000/health | python3 -m json.tool
```

### Prometheus and Grafana

`just services` starts Prometheus (localhost:9090) and Grafana (localhost:3000, configurable via `GRAFANA_PORT`) alongside Qdrant and PostgreSQL.

Prometheus scrapes two targets every 15 seconds:
- The app at `host.docker.internal:8000/metrics` (admin-gated, token from `ADMIN_TOKEN` via docker-compose)
- Qdrant at `qdrant:6333/metrics` (native, no auth)

Grafana ships with a pre-provisioned dashboard ("Geo-AI Book Assistant") showing request rate, latency percentiles, error rate, LLM generation duration, retrieval duration, and Qdrant operations.

### Structured logging

The app uses JSON structured logging in production (`just serve`) and plain text in dev (`just dev`).
Every API request gets a `X-Request-ID` header for log correlation.
Access logs include method, path, status, latency, and client IP.

### Log locations

| Log | Location | Rotation |
|-----|----------|----------|
| App structured logs | `logs/app/app.log` | RotatingFileHandler (10 MB x 5) |
| App raw stdout/stderr | `logs/app/stdout.log`, `stderr.log` | `just log-rotate` (>10 MB, keep 7) |
| Cloudflare tunnel | `logs/tunnel/tunnel.log`, `tunnel.error.log` | `just log-rotate` (>10 MB, keep 7) |
| Indexing scripts | `logs/scripts/indexing_*.log` | Pruned to 10 most recent |
| Transcript scripts | `logs/scripts/cleanup_*.log`, `transcription_*.log` | Pruned to 10 most recent |
| Docker containers | `docker logs <name>` | Docker driver (10 MB x 3) |
| DB backups / nightly | `backups/nightly.log` | Overwritten each run |

### Log rotation

Launchd logs (stdout, stderr, tunnel) are rotated by `just log-rotate`, which runs nightly alongside `just db-backup` via the `com.geoai.nightly` plist.
The recipe uses copytruncate: files over 10 MB are copied with a timestamp, gzipped, and the original is truncated.
No sudo required. Keeps 7 rotated copies per file.

App structured logs (`logs/app/app.log`) are rotated automatically by Python's RotatingFileHandler.
Script logs are pruned to 10 most recent per prefix on each run.

### Docker service logs

```bash
docker compose logs -f qdrant       # vector DB logs
docker compose logs -f postgres     # database logs
docker compose logs -f prometheus   # metrics collector logs
docker compose logs -f grafana      # dashboard logs
```

Check Qdrant dashboard at http://localhost:6333/dashboard for collection stats and indexed chunk counts.

## 10. Maintenance

### Database backups

Back up PostgreSQL on demand:

```bash
just db-backup
```

This writes a timestamped compressed dump to `backups/` (e.g. `backups/geoai_20261001_020000.sql.gz`).

To schedule nightly backups at 2 AM via launchd (recommended over cron - launchd catches up on missed runs after sleep or shutdown):

```bash
cp infra/launchd/com.geoai.nightly.plist ~/Library/LaunchAgents/
# Edit ~/Library/LaunchAgents/com.geoai.nightly.plist:
#   Replace REPO_PATH with your clone location (e.g. /Users/yourname/geoai-book-assistant)
#   Replace HOMEBREW_PREFIX with /opt/homebrew (Apple Silicon) or /usr/local (Intel)
#   Replace DOCKER_BIN with the output of: dirname $(which docker)

launchctl load ~/Library/LaunchAgents/com.geoai.nightly.plist
```

Logs go to `backups/nightly.log`.
To stop: `launchctl unload ~/Library/LaunchAgents/com.geoai.nightly.plist`.

Each backup run automatically prunes old files, keeping the 30 most recent.
At nightly frequency this gives 30 days of history with no manual cleanup.

To restore from a backup (replaces all data in the database):

```bash
just db-restore backups/geoai_20261001_020000.sql.gz
```

If the backup predates a schema change, run `just db-migrate` after restoring.

### Reset and cleanup

Reset the database without re-indexing (keeps Qdrant vectors intact):

```bash
just db-reset
```

Wipe everything for a completely fresh start (removes all volumes and data):

```bash
just clean
```
