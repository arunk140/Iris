# Development Plan — Project Iris (Phase 1)

## Implementation Order

Each step is designed to be independently verifiable before moving to the next.

```
Step 1:  Scaffold — mkdir, .env, .gitignore
Step 2:  db/init.sql — schema, indexes, pgVector
Step 3:  Watcher service — poll source/, rename, publish to RabbitMQ
Step 4:  Frame Extractor — consume video.new, extract frames, publish frames.extracted
Step 5:  Embedder — consume frames.extracted, CLIP inference, write to PostgreSQL
Step 6:  docker-compose.yml — wire everything together
Step 7:  End-to-end smoke test
```

---

## Step 1 — Scaffold

Create the directory tree and config files.

```
project-iris/
├── services/
│   ├── watcher/
│   ├── frame-extractor/
│   └── embedder/
├── db/
├── source/
├── frames/
├── .env
├── .gitignore
└── docker-compose.yml          (placeholder — filled in Step 6)
```

**Files**: `.env`, `.gitignore`

**Verification**: `ls -R` shows empty directory tree.

---

## Step 2 — Database Schema

Write `db/init.sql` with:
- `CREATE EXTENSION vector`
- `videos` table (UUID pk, filename, path, size, duration, width, height, fps, status, timestamps)
- `frames` table (BIGSERIAL pk, video_id FK, idx, timestamp_s, embedding vector(512))
- HNSW index on embedding
- B-tree indexes on `frames.video_id` and `videos.status`

**Files**: `db/init.sql`

**Verification**: `docker run pgvector/pgvector:pg16` and run `psql -f init.sql`.

---

## Step 3 — Watcher Service

A Python service that polls `SOURCE_DIR` for `*.mp4` files, assigns UUIDs, renames
them, and publishes messages to RabbitMQ.

**Files**:
- `services/watcher/pyproject.toml` — deps: `pika`, `health-check-http` (or stdlib http.server)
- `services/watcher/Dockerfile` — multi-stage with uv
- `services/watcher/watcher.py`

**Behaviour**:
- Poll loop with configurable interval
- Size-stability check (stat twice, 1 s apart)
- Atomic rename to `{uuid}.mp4`
- Publish to `iris.ingest` / `video.new`
- HTTP `/health` endpoint on port 8080
- Graceful SIGTERM handling

**Verification**: Run `docker compose up watcher`, drop a file into `source/`, see log line.

---

## Step 4 — Frame Extractor Service

Consumes `video.new` messages, opens video with OpenCV, extracts 1 frame/second,
encodes as JPEG base64, publishes batch to `frames.extracted`.

**Files**:
- `services/frame-extractor/pyproject.toml` — deps: `pika`, `opencv-python-headless`, `psycopg[binary]`
- `services/frame-extractor/Dockerfile`
- `services/frame-extractor/extractor.py`

**Behaviour**:
- Manual ACK only after successful publish
- Idempotency check via DB
- DEBUG mode writes frames to disk
- Batch splitting for large payloads (multiple messages)
- Handle corrupt frames gracefully (skip, log, continue)

**Verification**: Run full stack up to this point, drop a 10-second video, see frames
extracted and published.

---

## Step 5 — Embedding Service

Consumes `frames.extracted`, runs CLIP ViT-B/32 inference, writes to PostgreSQL.

**Files**:
- `services/embedder/pyproject.toml` — deps: `pika`, `opencv-python-headless`, `open-clip-torch`, `torch`, `psycopg[binary]`, `pgvector-python`
- `services/embedder/Dockerfile`
- `services/embedder/embedder.py`

**Behaviour**:
- Load CLIP model on startup
- Batch inference with sub-batching (64 frames) to avoid GPU OOM
- Re-assemble split batches (accumulate, flush on completion or TTL expiry)
- DB transaction: UPSERT `videos`, bulk INSERT `frames`
- Healthy checks: RabbitMQ, PostgreSQL, pgVector extension

**Verification**: Full pipeline — drop video in `source/`, wait, query PostgreSQL for
embeddings.

---

## Step 6 — docker-compose.yml

Wire all services + infra together.

**Services**:
- `rabbitmq:3.13-management`
- `postgres: pgvector/pgvector:pg16`
- `watcher` (build + depends_on)
- `frame-extractor` (build + depends_on)
- `embedder` (build + depends_on)

**Volumes**: `source/`, `frames/`, `pg_data`, `rabbitmq_data`, `model_cache`

**Healthchecks**: All infra services have healthchecks; services wait for them.

**GPU support**: Commented-out deploy section for embedder.

**Files**: `docker-compose.yml`

**Verification**: `docker compose up --build` starts everything cleanly.

---

## Step 7 — End-to-End Smoke Test

1. `docker compose up --build -d`
2. Wait for all health checks to pass.
3. `cp tests/sample_10s.mp4 source/`
4. Tail logs until "Stored N embeddings" appears.
5. Query DB to confirm:
   ```sql
   SELECT id, filename, duration_s, status FROM videos;
   SELECT COUNT(*) AS frames FROM frames;
   SELECT video_id, idx, timestamp_s, embedding::text FROM frames LIMIT 5;
   ```

---

## File Change Summary

| # | File | Action |
|---|------|--------|
| 1 | `.env` | Create |
| 2 | `.gitignore` | Create |
| 3 | `db/init.sql` | Create |
| 4 | `services/watcher/pyproject.toml` | Create |
| 5 | `services/watcher/Dockerfile` | Create |
| 6 | `services/watcher/watcher.py` | Create |
| 7 | `services/frame-extractor/pyproject.toml` | Create |
| 8 | `services/frame-extractor/Dockerfile` | Create |
| 9 | `services/frame-extractor/extractor.py` | Create |
| 10 | `services/embedder/pyproject.toml` | Create |
| 11 | `services/embedder/Dockerfile` | Create |
| 12 | `services/embedder/embedder.py` | Create |
| 13 | `docker-compose.yml` | Create |
