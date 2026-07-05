# Iris — Video Intelligence Platform

Auto-ingest raw video files, extract frames at a configurable rate (default 5 fps),
embed each frame with CLIP ViT-B/32 (512-dim), and explore them through an
interactive visualizer with similarity search and flow building.

```
source/ ──► Watcher ──► RabbitMQ ──► Frame Extractor ──► RabbitMQ ──► Embedder ──► PostgreSQL + pgVector
                                                                                         │
                                                                                    Visualizer (Streamlit)
```

## Quickstart

```bash
# Start everything
docker compose up --build -d

# Drop a video into source/
cp ~/clips/ride_001.mp4 source/

# Watch the pipeline
docker compose logs -f

# Open the visualizer at http://localhost:8501
```

## Pipeline

| Step | Service | What happens |
|---|---|---|
| 1 | **Watcher** | Polls `source/` every 10s for `.mp4`/`.mov` files. Renames to UUID, computes SHA-256 hash, publishes `video.new` to RabbitMQ |
| 2 | **Frame Extractor** | Consumes `video.new`. Opens video with OpenCV, seeks to each frame at `FRAME_RATE` intervals, writes JPEG to `frames/{video_id}/{idx:06d}.jpg`. Publishes frame paths and timestamps to `frames.extracted` |
| 3 | **Embedder** | Consumes `frames.extracted`. Loads CLIP ViT-B/32 model, embeds frames in batches, stores `(video_id, idx, timestamp_s, embedding)` in pgVector. Deduplicates by `content_hash` — skips frames from already-processed videos |
| 4 | **Visualizer** | Streamlit app that reads embeddings from DB, projects them into 2D for interactive exploration |

## Visualizer UI

The Streamlit interface is organized into tabs and collapsible sections:

### Sidebar

| Section | Contents |
|---|---|
| **Stats** | Videos total/done, frame count, queue depth, error count |
| **Visualization** | Dimensionality reduction method (PCA/t-SNE/UMAP), sample size slider, color-by selector |
| **Search** | Cross-video toggle — limits neighbors to one per other video |
| **Flow Settings** | Clip duration (1-10s), flow steps (0-500), transition overlap, crossfade toggle |
| **Advanced** | Compute t-SNE/UMAP button, data refresh button |

### Main tabs

| Tab | Contents |
|---|---|
| **Projection** | Interactive scatter plot. Click a point to select it — an expander below shows the frame image and metadata |
| **Neighbors** | 10 nearest neighbors for the selected point, each with frame thumbnail and cosine distance |
| **Flow** | Build Flow button to generate a clip chain starting from the selected point. Flow steps shown in a collapsible section. Merge button below to combine clips into a single video |

## Visualizer Features

- **Embedding projection** — PCA (instant, pre-computed once and stored in DB), t-SNE, and UMAP
- **Similarity search** — click a point → 10 nearest neighbors by cosine distance. Cross-video mode uses `DISTINCT ON (video_id)` to show one result per other video
- **Build Flow** — starting from any selected frame, chains clips across videos by finding the most similar frame in a different video at each step. Uses a dedup set to avoid repeating the same clip. Steps are collapsible with embedded video players
- **Merge** — two modes:
  - **Concat** (default): `ffmpeg -f concat -c copy` — stream copy, instant, no re-encode
  - **Crossfade**: binary merge tree using `xfade` filter with `ThreadPoolExecutor` — re-encodes (slow), produces smooth transitions
- **Stats panel** — live counts of videos, frames, and queue depth in the sidebar

## Configuration

See `.env` for defaults. Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_DSN` | `postgresql://iris:iris@postgres:5432/iris` | Database connection |
| `RABBITMQ_DSN` | `amqp://iris:iris@rabbitmq:5672/` | Message broker |
| `FRAME_RATE` | `5` | Frames extracted per second of video |
| `CACHE_DIR` | `/tmp/iris_cache` | Temporary clip and merge file storage |
| `FRAMES_DIR` | `/data/frames` | Frame JPEG disk storage |
| `EMBEDDING_MODEL` | `ViT-B-32` | CLIP model variant |
| `EMBEDDING_BATCH_SIZE` | `64` | Batch size for CLIP inference |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## GPU Acceleration

Uncomment the `deploy` block for the embedder service in `docker-compose.yml` to
enable CUDA acceleration for CLIP inference. This reduces embed time from ~50s
per 1500-frame video to ~5s.

## Database Schema

### `videos`

| Column | Type | Description |
|---|---|---|
| `id` | `UUID PRIMARY KEY` | Assigned by watcher |
| `filename` | `TEXT` | Original filename |
| `source_path` | `TEXT` | Full path to video file |
| `content_hash` | `TEXT` | SHA-256 of entire file (dedup key) |
| `size_bytes` | `BIGINT` | File size |
| `duration_s` | `DOUBLE PRECISION` | Video duration |
| `width` / `height` | `INT` | Video resolution |
| `original_fps` | `DOUBLE PRECISION` | Native frame rate |
| `status` | `TEXT` | `pending` → `processing` → `done` / `error` |
| `error_msg` | `TEXT` | Error details if failed |

### `frames`

| Column | Type | Description |
|---|---|---|
| `id` | `BIGSERIAL` | Auto-increment |
| `video_id` | `UUID` | FK to videos |
| `idx` | `INT` | Frame index (0, 1, 2…) within the video |
| `timestamp_s` | `DOUBLE PRECISION` | Exact timestamp in seconds |
| `embedding` | `vector(512)` | CLIP ViT-B/32 embedding |
| `pca_x` / `pca_y` | `DOUBLE PRECISION` | Pre-computed PCA coordinates |
| `UNIQUE (video_id, idx)` | — | Prevents duplicate frame entries |

Indexes: HNSW index on `embedding` for fast approximate nearest neighbor search.
