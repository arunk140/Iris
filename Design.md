# Project Iris — Design Spec

## Overview

Project Iris is a Video Intelligence Platform for processing unlabelled bike-helmet
footage. In Phase 1, raw video files dropped into a `source/` folder are automatically
detected, queued via RabbitMQ, decoded to frames at 1 frame per second, and embedded
using CLIP ViT-B/32 (512-dimension vectors). All embeddings and video metadata are
stored in PostgreSQL with the pgVector extension, enabling future similarity search,
clustering, and classification.

### Goals (Phase 1)

- Provide a fully automated ingestion pipeline: file appears → embeddings in DB.
- Support 40 GB of video (mixed durations, resolutions) without manual intervention.
- Store every frame's embedding alongside its source video ID and timestamp.
- Keep each pipeline stage in its own service so new stages can be inserted later.
- Run locally via docker-compose with minimal setup.

### Non-Goals (Phase 1)

- No video-level classification or object detection — that comes in later phases.
- No web UI or API — the database is the sole output.
- No audio processing — frames only.
- No distributed scaling beyond docker-compose replicas.

### Assumptions

- Videos are standard mp4/h264 files (handled by OpenCV).
- Each file is a self-contained video (no playlists, segments, or HLS).
- Files are moved into `source/` atomically (e.g., written to a temp dir then renamed
  in, or `rsync`-ed, or copied in one shot). Partial/corrupt writes during the poll
  window are possible; the watcher handles this by checking file size stability.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        docker-compose (iris-network)                  │
│                                                                      │
│  ┌──────────────┐   5672    ┌──────────────────┐   5432             │
│  │  RabbitMQ     │◄────────►│  PostgreSQL       │                    │
│  │  (iris-broker)│          │  + pgVector       │                    │
│  │  ports:       │          │  (iris-db)        │                    │
│  │   5672, 15672 │          │  port: 5432       │                    │
│  └──────┬───────┘          └────────▲──────────┘                    │
│         │                           │                                │
│  ┌──────▼───────┐          ┌────────┴──────────┐                    │
│  │  Watcher      │          │  Embedding Service │                    │
│  │  (poll)       │          │  (iris-embedder)   │                    │
│  │  service:     │          │  depends:          │                    │
│  │   iris-watcher│          │   rabbitmq, pg     │                    │
│  └──────┬───────┘          └────────────────────┘                    │
│         │                           ▲                                │
│         │ publish: video.new        │ consume: frames.extracted      │
│         │                           │                                │
│  ┌──────▼───────────────────────────┴────────┐                      │
│  │           Frame Extractor                  │                      │
│  │           (iris-frame-extractor)           │                      │
│  │           depends: rabbitmq                │                      │
│  └───────────────────────────────────────────┘                      │
│                                                                      │
│  ┌──────────────┐      ┌──────────────────┐                         │
│  │  source/      │      │  frames/          │                        │
│  │  (bind mount) │      │  (bind mount,     │                        │
│  │  /data/source │      │   DEBUG only)     │                        │
│  └──────────────┘      └──────────────────┘                         │
└──────────────────────────────────────────────────────────────────────┘
```

### Startup Order

1. `postgres` — must be healthy before any service that writes to it.
2. `rabbitmq` — must be healthy before any publisher/consumer.
3. `embedder` — starts first among workers (it declares exchanges/queues and waits for
   rabbitmq + postgres).
4. `frame-extractor` — depends on rabbitmq.
5. `watcher` — depends on rabbitmq; starts last.

RabbitMQ exchanges and queues are declared idempotently by the embedder (the first
consumer to start). All services use a connection retry loop with exponential backoff.

---

## Services

### 1. Watcher Service

- **Service name**: `iris-watcher`
- **Purpose**: Polls `source/` every N seconds for new files matching the watch
  extension. Assigns each file a UUID, renames it in-place, and publishes an
  ingestion message to RabbitMQ.

- **Startup**:
  - Connects to RabbitMQ with retry (exponential backoff, 30 s max).
  - Ensures the `source/` directory exists (creates if missing).

- **Poll loop**:
  1. List all entries in `SOURCE_DIR` matching `*{WATCH_EXT}` (case-insensitive).
  2. For each file:
     a. Check file size has stabilised (re-stat after 1 s; if size changed, skip
        until next poll — avoids picking up partially written files).
     b. Generate `uuid4()`.
     c. `os.rename(existing_path, f"{video_id}{WATCH_EXT}")` — this atomic rename
        removes the file from future polls (no longer matches the glob).
     d. Publish to `iris.ingest` / `video.new`.
     e. On publish failure: rename back to original name (so it is retried).
     f. Log success.
  3. Sleep `POLL_INTERVAL_S`.

- **Message payload** (`video.new`):
  ```json
  {
    "video_id": "a1b2c3d4-...",       // UUID v4
    "filename": "ride_001.mp4",    // original filename
    "path": "/data/source/a1b2c3d4-....mp4",
    "size_bytes": 2147483648,
    "detected_at": "2026-07-05T10:30:00Z"
  }
  ```

- **Error handling**:
  - RabbitMQ connection lost: log warning, retry connect with backoff, continue
    polling (files accumulate in source/).
  - Publish failure: rename file back to original, log error, retry next cycle.
  - Corrupt/unreadable files that cause rename failure: log and skip (alert via
    health check).

- **Graceful shutdown**:
  - Trap `SIGTERM` / `SIGINT`.
  - Finish current poll iteration (don't interrupt mid-rename).
  - Close RabbitMQ channel and connection.

- **Health check**: Exposes a lightweight HTTP endpoint on port 8080 — `GET /health`
  returns `{"status": "ok"}` or `{"status": "error", "reason": "..."}` if RabbitMQ
  is disconnected for more than `HEALTH_TIMEOUT_S`.

#### Edge Cases — Watcher

| Scenario | Behaviour |
|---|---|
| Empty `source/` | Poll loop runs, finds nothing, sleeps, repeats. |
| File is still being written | Size changes between two stats 1 s apart → skip. On next poll, size will likely be stable. |
| File with no video content | Renamed and queued anyway; `Frame Extractor` handles the failure. |
| Multiple files appear simultaneously | Processed sequentially in one poll iteration (avoids thundering herd on queue/disk). |
| File renamed externally after detection | `os.rename` will fail (file not found) → log warning, skip. |

---

### 2. Frame Extractor Service

- **Service name**: `iris-frame-extractor`
- **Purpose**: Consumes `video.new` messages, opens the video with OpenCV, extracts
  one frame per second of video, encodes each frame as a JPEG base64 string, and
  publishes the batch to the next queue.

- **Startup**:
  - Connects to RabbitMQ with retry.
  - Optionally connects to PostgreSQL (used only for idempotency check + setting
    video duration).

- **Consume loop**:
  1. Receive message from queue `iris.video.new`.
  2. **Idempotency**: Check if `videos.status != 'pending'` for this `video_id` in
     PostgreSQL. If already `done` or `processing`, `ACK` and discard.
  3. Set `videos.status = 'processing'` and record `duration_s`.
  4. Open video at `message.path` with `cv2.VideoCapture`.
  5. Query `CAP_PROP_FRAME_COUNT` and `CAP_PROP_FPS` → compute total duration.
  6. For each integer second `t` from 0 to floor(duration):
     - Seek to `t * original_fps` frames with `cv2.CAP_PROP_POS_FRAMES`.
     - Read frame.
     - Convert BGR → RGB.
     - Encode JPEG bytes in-memory (no disk write unless `DEBUG`).
     - Base64-encode the JPEG bytes.
     - Append to frames list and timestamps list.
  7. After extraction (some frames may be empty near end — handle by skipping):
     - If zero frames extracted: set `status = 'error'`, `NACK` with requeue=false,
       publish to dead-letter.
     - Publish batch message to `iris.processing` / `frames.extracted`.
     - On publish success: `ACK` the original message.
     - On publish failure: keep the message un-acked (will be requeued by RabbitMQ
       after consumer timeout).
  8. Release `cv2.VideoCapture`.

- **Message payload** (`frames.extracted`):

  ```json
  {
    "video_id": "a1b2c3d4-...",
    "filename": "ride_001.mp4",
    "frames": [
      "/9j/4AAQ...",   // base64-encoded JPEG, frame 0
      "/9j/4AAQ..."    // base64-encoded JPEG, frame 1
    ],
    "timestamps": [0.0, 1.0, 2.0, ...],
    "fps_original": 29.97,
    "duration_s": 30.0,
    "width": 1920,
    "height": 1080,
    "total_frames": 30,
    "extracted_at": "2026-07-05T10:31:00Z"
  }
  ```

- **DEBUG mode**:
  - When `DEBUG=true`, after extracting each frame, also write the JPEG to
    `{FRAMES_DIR}/{video_id}/{timestamp_s}.jpg` before base64-encoding.

- **Error handling**:
  - Video file not found / corrupt / unreadable: set `status = 'error'` with error
    detail, `NACK` with requeue=false, send to `iris.dlx` with reason header.
  - OpenCV seek failure: log the frame index that failed, continue with next.
  - Frames payload too large for RabbitMQ (default ~128 MB limit): split the batch
    into multiple messages. Each sub-message has `"batch_index": i` and
    `"total_batches": n`. The embedder re-assembles on the other side using a
    Redis or in-memory buffer (see Embedder section).
  - RabbitMQ disconnect during publish: message remains un-acked → requeued →
    retried by another consumer instance.
  - Out of memory: process-level OOM kill → container restarts → un-acked messages
    requeue.

- **Graceful shutdown**:
  - Trap `SIGTERM`.
  - Stop consuming new messages (`basic_cancel`).
  - Finish extracting the current video (do not drop it).
  - Close OpenCV and RabbitMQ.

- **Health check**: HTTP `GET /health` on port 8080 — checks RabbitMQ connection
  and (optionally) that PostgreSQL is reachable.

#### Edge Cases — Frame Extractor

| Scenario | Behaviour |
|---|---|
| Video is 0 seconds / cannot open | Zero frames → `status = error`, NACK to DLX. |
| Single-frame video (e.g., image file renamed to .mp4) | Extracts frame 0, publishes single-frame batch. Valid. |
| Variable frame rate video | `CAP_PROP_FPS` returns average; seeking by frame index may drift slightly from real time. Acceptable for Phase 1. |
| Frames payload > RabbitMQ limit | Automatic splitting into multiple messages. |
| Video file deleted before processing | `cv2.VideoCapture` fails → set error status and DLX. |

---

### 3. Embedding Service

- **Service name**: `iris-embedder`
- **Purpose**: Consumes `frames.extracted` messages, decodes frames, embeds each
  frame with CLIP ViT-B/32 (via `open_clip`), and writes video metadata + frame
  embeddings to PostgreSQL.

- **Startup**:
  1. Connect to RabbitMQ and PostgreSQL (both with retry).
  2. Load CLIP model (`open_clip.create_model_and_transforms('ViT-B-32')`) and
     tokenizer. Model weights are downloaded on first run and cached in a volume.
  3. Declare exchanges and queues (idempotent) — since this is the terminal stage,
     it owns the topology declaration.
  4. Run schema migrations (execute `init.sql` if tables don't exist).

- **Consume loop**:
  1. Receive message from `iris.frames.extracted`.
  2. **Idempotency**: Check if `videos.id = video_id` and `videos.status = 'done'`.
     If already done, `ACK` and skip.
  3. Decode each base64 frame back to a JPEG bytes object, then to a numpy array
     (RGB, uint8) with OpenCV or PIL.
  4. Apply CLIP preprocessing transforms (resize to 224×224, normalize).
  5. **Batched inference**: Stack frames into a single tensor of shape
     `[N, 3, 224, 224]` and run one `model.encode_image()` call.
  6. **Normalize**: L2-normalise each embedding vector.
  7. DB transaction:
     ```sql
     BEGIN;
     INSERT INTO videos (id, filename, source_path, size_bytes, duration_s, status)
     VALUES (...) ON CONFLICT (id) DO UPDATE SET status = 'done' WHERE videos.status = 'processing';
     -- (skip insert if already done)
     INSERT INTO frames (video_id, idx, timestamp_s, embedding)
     VALUES ...  -- bulk insert, all rows in one statement
     ON CONFLICT (video_id, idx) DO NOTHING;
     COMMIT;
     ```
  8. `ACK` the RabbitMQ message only after the transaction commits.
  9. Log completion (video_id, frame count, duration).

- **Batch splitting (large videos)**:
  - If the message has `total_batches > 1`, the embedder accumulates frames in a
    dict keyed by `video_id` (a `dict[str, list[tuple[int, float, bytes]]]`).
  - After receiving all batches (detected by `batch_index + 1 == total_batches`),
    it sorts by `batch_index`, concatenates frames in order, and processes the
    full set.
  - A TTL (5 minutes, configurable via `BATCH_TTL_S`) guards against lost messages:
    if not all batches arrive within the window, the partial data is discarded
    and the video is marked `error`.

- **Error handling**:
  - DB connection lost: message remains un-acked, consumer retries after reconnect.
  - Embedding inference error (OOM, bad frame data): NACK with requeue=false, DLX.
  - Duplicate message: upsert handles it; `ON CONFLICT DO NOTHING` on frames.
  - Corrupt frame bytes: skip that single frame (log warning), continue with the
    rest. Frame count will be less than expected.

- **GPU support**:
  - If `torch.cuda.is_available()` returns true, the model and tensors are moved
    to CUDA. The container must be run with `--gpus all` or use the `nvidia` runtime.
  - No GPU: falls back to CPU. A 30-second video (~30 frames) takes ~2–3 seconds on
    CPU; 40 GB of video will process but slowly. GPU is strongly recommended.

- **Graceful shutdown**:
  - Trap `SIGTERM`.
  - Finish the current batch (complete the DB write).
  - Unload model from GPU if applicable.
  - Close DB and RabbitMQ connections.

- **Health check**: HTTP `GET /health` on port 8080 — checks RabbitMQ, PostgreSQL
  connectivity, and that the model is loaded. Also exposes `GET /ready` which
  additionally checks pgVector extension is installed.

#### Edge Cases — Embedding Service

| Scenario | Behaviour |
|---|---|
| Same video arrives twice | `ON CONFLICT` handles it; second message is ACKed after idempotency check. |
| GPU OOM during inference (video too long) | If the batch (all frames of one video) exceeds GPU memory, split into sub-batches of 64 frames internally before the model call. |
| Model download fails on first run | The embedder container will fail to start. The model weights volume is persisted, so a restart picks up where it left off. |
| PostgreSQL is down at startup | Retry loop with backoff — service stays unhealthy until DB is reachable. |

---

## Database Schema (PostgreSQL + pgVector)

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE videos (
    id          UUID PRIMARY KEY,
    filename    TEXT NOT NULL,
    source_path TEXT NOT NULL,
    size_bytes  BIGINT,
    duration_s  DOUBLE PRECISION,
    width       INT,
    height      INT,
    original_fps DOUBLE PRECISION,
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending', 'processing', 'done', 'error')),
    error_msg   TEXT,                         -- populated when status = 'error'
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE frames (
    id          BIGSERIAL PRIMARY KEY,
    video_id    UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    idx         INT NOT NULL,                 -- 0-based frame index
    timestamp_s DOUBLE PRECISION NOT NULL,
    embedding   vector(512) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (video_id, idx)
);

-- HNSW index for approximate nearest-neighbour search on cosine distance
CREATE INDEX idx_frames_embedding ON frames
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- Index for fetching frames by video (common query pattern)
CREATE INDEX idx_frames_video_id ON frames (video_id);

-- Index for status-based queries (monitoring / dashboard)
CREATE INDEX idx_videos_status ON videos (status);
```

### Design Notes

- **`vector(512)`** matches CLIP ViT-B/32 output dimensionality.
- **HNSW index** (`m=16`, `ef_construction=200`) is a good default for datasets under
  10 M rows. Tune `ef_search` at query time via `SET hnsw.ef_search = 100;`.
- **`ON DELETE CASCADE`** on `videos.id` — deleting a video removes all its frames.
- **Status check constraint** prevents invalid states.
- **`error_msg`** captures why a video failed, for debugging.

### Migration Strategy

- Phase 1 uses raw SQL files applied at embedder startup.
- If the schema becomes complex in later phases, adopt Alembic for versioned
  migrations.

---

## RabbitMQ Topology

### Exchanges and Queues

| Exchange            | Type   | Routing Key         | Queue                      | Consumer            | TTL (message) | Max length |
|---------------------|--------|---------------------|----------------------------|---------------------|---------------|------------|
| `iris.ingest`       | direct | `video.new`         | `iris.video.new`           | Frame Extractor     | 1 hour        | 1000       |
| `iris.processing`   | direct | `frames.extracted`  | `iris.frames.extracted`    | Embedding Service   | 1 hour        | 500        |
| `iris.dlx`          | fanout | —                   | `iris.dlq`                 | (manual inspection) | —             | —          |

### Dead-Letter Configuration

Each primary queue is configured with:

```
arguments:
  x-dead-letter-exchange:    iris.dlx
  x-dead-letter-routing-key: ""          (routing key discarded; DLX is fanout)
  x-message-ttl:             3600000     (1 hour, ms)
  x-max-length:              1000        (safety cap)
  x-overflow:                reject-publish-dlx
```

When a consumer NACKs with `requeue=false` or a message TTL expires, the message is
published to `iris.dlx` for manual inspection / re-queue via a management tool.

### Consumer Configuration

| Queue                    | Prefetch | Auto-ACK | Consumer timeout |
|--------------------------|----------|----------|------------------|
| `iris.video.new`         | 1        | false    | 30 min           |
| `iris.frames.extracted`  | 1        | false    | 30 min           |

- **Prefetch = 1**: each consumer processes one message at a time. This prevents
  a single consumer from pulling many large messages and running out of memory.
- **Auto-ACK = false**: all ACKs are manual, sent only after the work succeeds.
- **Consumer timeout**: if a consumer does not ACK within 30 minutes, RabbitMQ
  considers it dead and re-queues the message. This covers crashes and hangs.

### Topology Declaration

The Embedding Service declares all exchanges and queues on startup (idempotent
declarations). The Frame Extractor and Watcher only declare the exchange they
publish to. This avoids duplicate declaration conflicts.

```python
# Pseudo-code for topology setup (embedder startup)
channel.exchange_declare('iris.ingest',     exchange_type='direct', durable=True)
channel.exchange_declare('iris.processing', exchange_type='direct', durable=True)
channel.exchange_declare('iris.dlx',        exchange_type='fanout', durable=True)

channel.queue_declare('iris.video.new',        durable=True, arguments=dlx_args)
channel.queue_declare('iris.frames.extracted', durable=True, arguments=dlx_args)
channel.queue_declare('iris.dlq',              durable=True)

channel.queue_bind('iris.video.new',        'iris.ingest',     routing_key='video.new')
channel.queue_bind('iris.frames.extracted', 'iris.processing', routing_key='frames.extracted')
channel.queue_bind('iris.dlq',              'iris.dlx')
```

---

## Directory Layout

```
project-iris/
│
├── docker-compose.yml              # All services + infra
├── .env                            # Environment variable defaults (committed)
├── .gitignore
├── README.md                       # Quickstart
│
├── services/
│   ├── watcher/
│   │   ├── Dockerfile              # Multi-stage: uv install → app
│   │   ├── pyproject.toml          # Dependencies: pika, aiohttp (health)
│   │   ├── uv.lock
│   │   └── watcher.py              # Entrypoint (~150 lines)
│   │
│   ├── frame-extractor/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml          # Dependencies: pika, opencv-python-headless, psycopg
│   │   ├── uv.lock
│   │   └── extractor.py            # Entrypoint (~250 lines)
│   │
│   ├── embedder/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml          # Dependencies: pika, opencv-python-headless, open_clip_torch,
│   │   │                           #                torch, psycopg, pgvector-python
│   │   ├── uv.lock
│   │   └── embedder.py             # Entrypoint (~300 lines)
│   │
│   └── pipeline-example/           # Template for adding new pipeline stages
│       ├── Dockerfile
│       ├── pyproject.toml
│       ├── uv.lock
│       └── pipeline_example.py     # Logs incoming messages (hello world)
│
├── db/
│   └── init.sql                    # Schema + extensions
│
├── source/                         # Drop videos here (bind mount, created by docker-compose)
└── frames/                         # DEBUG frames (bind mount, created only if DEBUG=true)
```

### Dockerfile Pattern (all services)

```dockerfile
FROM python:3.12-slim AS base

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-cache

COPY *.py ./

FROM base AS runner
EXPOSE 8080
CMD ["uv", "run", "python", "watcher.py"]
```

Rationale for multi-stage: the first stage installs dependencies and freezes them;
the runner stage is the runtime image. The `uv sync` output (`/app/.venv`) is used
by `uv run`, which automatically activates the virtual environment.

---

## Configuration (Environment Variables)

| Variable              | Default                                | Services                  | Description |
|-----------------------|----------------------------------------|---------------------------|-------------|
| `RABBITMQ_HOST`       | `rabbitmq`                             | all                       | RabbitMQ hostname |
| `RABBITMQ_PORT`       | `5672`                                 | all                       | RabbitMQ AMQP port |
| `RABBITMQ_VHOST`      | `/`                                    | all                       | Virtual host |
| `RABBITMQ_USER`       | `guest`                                | all                       | Username |
| `RABBITMQ_PASS`       | `guest`                                | all                       | Password |
| `POSTGRES_DSN`        | `postgresql://iris:iris@pg:5432/iris`  | frame-extractor, embedder | PostgreSQL connection string |
| `SOURCE_DIR`          | `/data/source`                         | watcher                   | Watch directory (bind mount) |
| `WATCH_EXT`           | `.mp4`                                 | watcher                   | File extension to detect |
| `POLL_INTERVAL_S`     | `10`                                   | watcher                   | Seconds between polls |
| `DEBUG`               | `false`                                | frame-extractor           | If true, write frames to disk |
| `FRAMES_DIR`          | `/data/frames`                         | frame-extractor           | Directory for DEBUG frame dumps |
| `EMBEDDING_MODEL`     | `ViT-B-32`                             | embedder                  | CLIP model variant |
| `EMBEDDING_DIM`       | `512`                                  | embedder                  | Dimensionality (must match model) |
| `EMBEDDING_BATCH_SIZE`| `64`                                   | embedder                  | Max frames per inference batch (to avoid GPU OOM) |
| `BATCH_TTL_S`         | `300`                                  | embedder                  | Seconds to wait for all split batches |
| `CONN_RETRY_DELAY_S`  | `1`                                    | all                       | Initial retry delay (exponential backoff) |
| `CONN_MAX_RETRIES`    | `0` (infinite)                         | all                       | Max connection retries (0 = forever) |
| `HEALTH_PORT`         | `8080`                                 | all                       | Health check HTTP port |
| `LOG_LEVEL`           | `INFO`                                 | all                       | Python logging level |

---

## Data Flow (Phase 1)

```
source/ride_001.mp4
    │
    │  [POLL: watcher wakes every 10 s]
    │
    ▼
┌─────────────────┐
│   Watcher        │
│                  │
│  1. Stat file    │──── size stable?
│  2. Generate UUID│      yes → continue
│  3. Rename to    │        no → skip, retry next cycle
│     {uuid}.mp4   │
│  4. Publish to   │
│     video.new    │
└────────┬────────┘
         │  exchange: iris.ingest / routing_key: video.new
         ▼
┌─────────────────┐
│   RabbitMQ       │
│   queue:         │
│   iris.video.new │
└────────┬────────┘
         │
         ▼
┌──────────────────────────┐
│   Frame Extractor         │
│                           │
│  1. Receive message       │
│  2. Idempotency check     │
│  3. Open video (OpenCV)   │
│  4. For t in 0..duration: │
│       seek(t)             │
│       read frame          │
│       encode JPEG         │
│       base64              │
│       [DEBUG → write disk]│
│  5. Publish to            │
│     frames.extracted      │
│  6. ACK original message  │
└────────┬──────────────────┘
         │  exchange: iris.processing / routing_key: frames.extracted
         ▼
┌─────────────────┐
│   RabbitMQ       │
│   queue:         │
│   frames.extracted│
└────────┬────────┘
         │
         ▼
┌──────────────────────────┐
│   Embedding Service       │
│                           │
│  1. Receive message       │
│  2. Idempotency check     │
│  3. Decode base64 frames  │
│  4. CLIP preprocessing    │
│  5. Batch inference       │
│     (sub-batches of 64)   │
│  6. L2-normalize vectors  │
│  7. DB transaction:       │
│     UPSERT videos         │
│     INSERT frames         │
│  8. ACK message           │
└──────────────────────────┘
         │
         ▼
┌─────────────────┐
│   PostgreSQL     │
│   + pgVector     │
│                  │
│   videos:        │
│     status=done  │
│   frames:        │
│     30 rows ×    │
│     512-dim vec  │
└─────────────────┘
```

### Error Paths

```
Any stage fails → message NACK'd (no requeue) → iris.dlx → iris.dlq
  │
  ├── Manual re-queue via RabbitMQ management UI (after fixing the issue)
  │
  └── Or: an automated retry service re-publishes from dlq with a delay
      (future phase)
```

---

## Extensibility (Future Phases)

The architecture supports adding new pipeline stages without modifying existing
services. Each new stage simply binds to the relevant exchange, declares its own
queue, and publishes results to a new routing key.

### Hello World: Pipeline Example

A reference template lives at `services/pipeline-example/`. It consumes messages
from a configurable exchange/routing key, logs them, and optionally publishes a
result. Uncomment the `pipeline-example` service in `docker-compose.yml` and
run `docker compose up -d` to see it in action:

```bash
docker compose logs -f pipeline-example
# 2026-07-05 ... Received message from iris.ingest: routing_key=video.new
```

To build your own stage: copy `pipeline-example/`, rename it, replace the log
statement with your actual logic, and pick a new exchange/routing key.

### Example: Audio Transcription

```
[Watcher] ──► video.new ──► [Frame Extractor] ──► frames.extracted ──► [Embedder]
                              │
                              └──► [Audio Extractor]  (new service)
                                        │
                                        └──► transcript.ready ──► [Search Indexer]
```

- `Audio Extractor` consumes `video.new`, extracts audio track with ffmpeg, runs
  Whisper, publishes transcripts.
- No changes needed to any existing service.

### Example: Object Detection

```
                        frames.extracted
                              │
     ┌────────────────────────┼────────────────────────┐
     ▼                        ▼                        ▼
[Embedder]           [Object Detector (YOLO)]   [Tracking Service]
                        │
                        └──► detections.ready ──► [Aggregator]
```

- `Object Detector` consumes the same `frames.extracted` stream as `Embedder`.
- Detections are stored in a new `detections` table.
- The HNSW index on embeddings is already in place for similarity queries.

### Example: Video-Level Aggregation

```
frames.extracted ──► [Embedder] ──► frames table (per-frame vectors)
                    │
                    └──► [Video Embedder]  (new service: mean-pool or attention pool)
                              │
                              └──► video_embeddings table
```

### Example: Feedback Loop / Re-indexing

If a model is fine-tuned on the collected embeddings, a re-indexing workflow can:

1. Truncate the `frames` table for affected videos.
2. Re-publish `video.new` messages for those videos (or directly enqueue into
   `frames.extracted` with the new frames).
3. The existing pipeline processes them idempotently.

### Example: Monitoring / Dashboard

```
iris.dlx ──► [Alert Service] ──► Slack / PagerDuty
```

A lightweight service tails `iris.dlq` and sends alerts if failure rates exceed a
threshold.

---

## Python Dependency Management (uv)

All services use [uv](https://docs.astral.sh/uv/) for Python version management,
virtual environments, and dependency resolution.

### Workflow

```bash
# Create a project and add dependencies
uv init watcher/
uv add pika
uv add aiohttp

# Sync with lockfile
uv lock

# Run the service
uv run python watcher.py

# Add a new dependency later
uv add opencv-python-headless
uv lock
```

### Docker Integration

In Dockerfiles, `uv sync --frozen` installs exact versions from the lockfile.
The `.venv` directory created by `uv sync` is used by `uv run`, eliminating any
need for system-level Python package management.

### Why uv over pip + venv

- **10–100× faster** dependency resolution.
- **`uv.lock`** is a deterministic, cross-platform lockfile — avoids `pip freeze`
  drift.
- **Single binary** — no need to install pip, setuptools, or wheel in the Docker
  image (except the Python runtime itself).
- **Workspaces** — if the services later share a common library (`iris-core`),
  uv workspaces handle it natively.

---

## Logging

All services use Python's `logging` module with structured JSON output when
`LOG_FORMAT=json` is set (default: plain text for local dev).

```
2026-07-05 10:30:00,123 [INFO] (iris-watcher) Processed file: video_id=a1b2c3..., filename=ride_001.mp4
2026-07-05 10:30:05,456 [INFO] (iris-frame-extractor) Extracted 30 frames from a1b2c3...
2026-07-05 10:30:08,789 [INFO] (iris-embedder) Stored 30 embeddings for a1b2c3... (duration=30.0s, batch_time=1.2s)
```

Key events:
- `Processed file` — watcher published a new video
- `Extracted N frames` — frame extractor finished
- `Stored N embeddings` — embedder committed to DB
- `Message NACK'd to DLX` — failure with reason
- `Connection lost, reconnecting...` — infra issue
- `Health check failed: ...` — service unhealthy

---

## Running (Phase 1)

### Prerequisites

- Docker Engine 24+ with Compose V2 plugin
- At least 8 GB RAM allocated to Docker (16 GB recommended)
- (Optional) NVIDIA Container Toolkit for GPU acceleration

### Quickstart

```bash
# 1. Clone / enter the project directory
cd project-iris

# 2. Start all services
docker compose up --build -d

# 3. Watch logs
docker compose logs -f

# 4. Drop a video into the source folder
cp ~/clips/ride_001.mp4 source/

# 5. Verify in logs — within ~30 seconds you should see:
#    iris-watcher-1          | Processed file: video_id=...
#    iris-frame-extractor-1  | Extracted 30 frames from ...
#    iris-embedder-1         | Stored 30 embeddings for ...

# 6. Query the database to confirm
docker compose exec -T pg psql -U iris -d iris -c "
  SELECT id, filename, duration_s, status FROM videos;
  SELECT video_id, COUNT(*) AS frames FROM frames GROUP BY video_id;
"
```

### Health Checks

```bash
# Each service exposes /health on port 8080
curl http://localhost:8081/health   # watcher (mapped port)
curl http://localhost:8082/health   # frame-extractor
curl http://localhost:8083/health   # embedder
```

Port mapping in docker-compose (optional, for debugging):

```yaml
services:
  watcher:
    ports: ["8081:8080"]
  frame-extractor:
    ports: ["8082:8080"]
  embedder:
    ports: ["8083:8080"]
```

### Debug Mode

```bash
DEBUG=true docker compose up -d
# Frames written to ./frames/{video_id}/ (bind mount)
```

### Tearing Down

```bash
docker compose down -v            # destroys volumes (DB data, frames)
docker compose down                # keeps volumes
```

### docker-compose.yml Structure

```yaml
services:
  rabbitmq:
    image: rabbitmq:3.13-management
    ports: ["5672:5672", "15672:15672"]
    volumes: ["rabbitmq_data:/var/lib/rabbitmq"]
    healthcheck: { test: ["CMD", "rabbitmq-diagnostics", "check_port_connectivity"], interval: 10s }

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: iris
      POSTGRES_PASSWORD: iris
      POSTGRES_DB: iris
    volumes:
      - pg_data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports: ["5432:5432"]
    healthcheck: { test: ["CMD-SHELL", "pg_isready -U iris"], interval: 5s }

  watcher:
    build: ./services/watcher
    volumes: ["${SOURCE_DIR:-./source}:/data/source"]
    environment:
      SOURCE_DIR: /data/source
      RABBITMQ_HOST: rabbitmq
    depends_on: { rabbitmq: { condition: service_healthy } }
    ports: ["8081:8080"]

  frame-extractor:
    build: ./services/frame-extractor
    volumes:
      - "${SOURCE_DIR:-./source}:/data/source"
      - "${FRAMES_DIR:-./frames}:/data/frames"
    environment:
      RABBITMQ_HOST: rabbitmq
      POSTGRES_DSN: postgresql://iris:iris@postgres:5432/iris
      DEBUG: ${DEBUG:-false}
    depends_on:
      rabbitmq: { condition: service_healthy }
      postgres: { condition: service_healthy }
    ports: ["8082:8080"]

  embedder:
    build: ./services/embedder
    environment:
      RABBITMQ_HOST: rabbitmq
      POSTGRES_DSN: postgresql://iris:iris@postgres:5432/iris
    volumes:
      - model_cache:/root/.cache/  # CLIP weights persist across restarts
    depends_on:
      rabbitmq: { condition: service_healthy }
      postgres: { condition: service_healthy }
    ports: ["8083:8080"]
    # Uncomment for GPU:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

volumes:
  rabbitmq_data:
  pg_data:
  model_cache:
```
