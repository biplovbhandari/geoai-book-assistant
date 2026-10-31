# Geo-AI Book Assistant
# Run `just --list` to see all available recipes.

# === macOS: native app, Docker services only ====================
# The app runs natively on macOS to access the Metal/MPS GPU.
# Docker runs only Qdrant and PostgreSQL (the `services` recipe).
# Workflow: just services -> just db-migrate -> just dev (or just serve)

# Start backing services and the dev server with hot-reload
dev: services
    uv run uvicorn api.main:app --reload --timeout-graceful-shutdown 30

# Run the app in production mode (no hot-reload, binds to all interfaces)
serve: services
    uv run uvicorn api.main:app --host 0.0.0.0 --port ${APP_PORT:-8000} --timeout-graceful-shutdown 30

# Run lint, type check, and tests
check: lint typecheck test

# Auto-fix formatting
format:
    uv run ruff format .

# ------------------ Quality checks -------------------------------

# Run ruff linter, format check, and CSS linter
lint:
    uv run ruff check .
    uv run ruff format --check .
    npx stylelint "**/*.css"

# Run pyright type checker
typecheck:
    uv run pyright

# Run the test suite
test *args='':
    uv run pytest tests/ -v {{ args }}

# ------------------ Database --------------------------------------

# Apply pending Alembic migrations
db-migrate:
    uv run alembic upgrade head

# Create a new auto-generated migration
db-revision msg:
    uv run alembic revision --autogenerate -m "{{ msg }}"

# Dump PostgreSQL to a timestamped compressed file in backups/
db-backup: (_db-backup-prune)
    #!/usr/bin/env bash
    set -euo pipefail
    if ! docker exec geoai-db pg_isready -U geoai -d geoai >/dev/null 2>&1; then
        echo "Error: PostgreSQL is not running" >&2
        exit 1
    fi
    mkdir -p backups
    outfile="backups/geoai_$(date +%Y%m%d_%H%M%S).sql.gz"
    docker exec geoai-db pg_dump -U geoai geoai | gzip > "$outfile"
    echo "Backup: $outfile ($(du -h "$outfile" | cut -f1))"

# Remove old backups, keeping the 30 most recent
_db-backup-prune:
    #!/usr/bin/env bash
    set -euo pipefail
    [[ -d backups ]] || exit 0
    count=$(ls -1 backups/geoai_*.sql.gz 2>/dev/null | wc -l)
    if (( count > 30 )); then
        ls -1t backups/geoai_*.sql.gz | tail -n +"31" | xargs rm -f
        echo "Pruned $((count - 30)) old backup(s)"
    fi

# Rotate launchd log files using copytruncate (no sudo, no restart needed)
log-rotate:
    #!/usr/bin/env bash
    set -euo pipefail
    max_bytes=$((10 * 1024 * 1024))  # 10 MB
    keep=7
    for logfile in logs/app/stdout.log logs/app/stderr.log logs/tunnel/tunnel.log logs/tunnel/tunnel.error.log; do
        [[ -f "$logfile" ]] || continue
        size=$(stat -f%z "$logfile" 2>/dev/null || echo 0)
        if (( size > max_bytes )); then
            rotated="${logfile}.$(date +%Y%m%d_%H%M%S)"
            cp "$logfile" "$rotated"
            : > "$logfile"
            gzip "$rotated"
            echo "Rotated $logfile ($((size / 1024 / 1024)) MB)"
        fi
    done
    # Prune old rotated copies
    for logfile in logs/app/stdout.log logs/app/stderr.log logs/tunnel/tunnel.log logs/tunnel/tunnel.error.log; do
        ls -1t "${logfile}".*.gz 2>/dev/null | tail -n +"$((keep + 1))" | xargs rm -f 2>/dev/null || true
    done

# Restore PostgreSQL from a backup file (replaces current data)
db-restore file:
    #!/usr/bin/env bash
    set -euo pipefail
    if [[ ! -f "{{ file }}" ]]; then
        echo "File not found: {{ file }}" >&2
        exit 1
    fi
    echo "WARNING: This will replace all data in the geoai database."
    read -p "Continue? [y/N] " -n 1 -r; echo
    [[ $REPLY =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 1; }
    docker exec geoai-db dropdb -U geoai --force --if-exists geoai
    docker exec geoai-db createdb -U geoai geoai
    gunzip -c "{{ file }}" | docker exec -i geoai-db psql -U geoai -d geoai --quiet
    echo "Restored from: {{ file }}"
    echo "Run 'just db-migrate' if the backup predates a schema change."

# ------------------ Content ---------------------------------------

# Index book chapters, PDFs, and transcripts into Qdrant (macOS: native, Metal GPU)
index *args='':
    BOOK_COMMIT_SHA=$(git -C book rev-parse HEAD) uv run python scripts/index_book.py {{ args }}

# Index using Docker indexer (Linux: CUDA GPU, uses GHCR image)
index-prod *args='':
    BOOK_COMMIT_SHA=$(git -C book rev-parse HEAD) docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile build run --rm indexer {{ args }}

# Transcribe YouTube lecture videos
transcribe *args='':
    uv run --group indexer python scripts/transcribe.py {{ args }}

# Clean transcripts (apply corrections from data/transcript_corrections.yml)
clean-transcript *args='':
    uv run python scripts/clean_transcript.py {{ args }}

# Render the book with the chat widget and copy to _book/ (local build)
render-book:
    docker compose --profile build run --rm quarto-builder
    mkdir -p _book
    docker compose --profile build run --rm book-copy

# Render the book using GHCR image (production deployment)
render-book-prod:
    docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile build run --rm quarto-builder
    mkdir -p _book
    docker compose --profile build run --rm book-copy

# === Linux: everything in Docker with NVIDIA GPU ================
# The app runs in Docker with CUDA GPU passthrough.
# Requires nvidia-container-toolkit. These recipes do NOT work on macOS.
# Workflow: just up (or up-prod) -> just render-book-prod -> just index-prod

# === Docker services (shared) ====================================

# Start Qdrant and PostgreSQL in the background (used by macOS recipes above)
services:
    docker compose up qdrant postgres prometheus grafana -d
    @echo "Waiting for PostgreSQL..."
    @until docker exec geoai-db psql -U geoai -d geoai -c "SELECT 1" >/dev/null 2>&1; do sleep 1; done
    @echo "PostgreSQL ready."

# Start the full Docker stack (local build, Linux with NVIDIA GPU)
up:
    docker compose --profile gpu up -d

# Start the full Docker stack using GHCR images (Linux with NVIDIA GPU)
up-prod:
    docker compose --profile gpu -f docker-compose.yml -f docker-compose.prod.yml up -d

# Start the full Docker stack with hot-reload (Linux dev)
dev-docker:
    docker compose --profile gpu -f docker-compose.yml -f docker-compose.dev.yml up

# Stop all Docker services
down:
    docker compose down

# Reset the database only (keeps Qdrant index intact)
db-reset:
    docker compose stop postgres
    rm -rf .data/postgres
    docker compose up postgres -d
    @echo "Waiting for PostgreSQL..."
    @until docker exec geoai-db psql -U geoai -d geoai -c "SELECT 1" >/dev/null 2>&1; do sleep 1; done
    @echo "PostgreSQL ready."
    uv run alembic upgrade head

# Stop services, remove volumes and data for a fresh start
clean:
    docker compose down -v --remove-orphans
    -docker rm -f $(docker ps -aq --filter "label=com.docker.compose.project=geoai-book-assistant") 2>/dev/null
    -docker volume ls -q --filter "name=geoai-book-assistant" | xargs -r docker volume rm 2>/dev/null
    rm -rf .data/qdrant .data/postgres _book

# Rebuild Docker images
build:
    docker compose build app quarto-builder indexer
