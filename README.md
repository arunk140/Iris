# Iris — Video Intelligence Platform

Auto-ingest raw video files, extract one frame per second, embed each frame with
CLIP ViT-B/32 (512-dim), and explore them through an interactive visualizer with
similarity search and flow building.

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

# Open the visualizer
# http://localhost:8501
```

## Services

| Service | Role |
|---|---|
| **Watcher** | Polls `source/` for `.mp4`/`.mov` files, assigns UUID, SHA-256 hash, publishes to `iris.ingest` |
| **Frame Extractor** | Consumes `video.new`, extracts 1 frame/sec via OpenCV, writes JPEGs to disk, publishes paths |
| **Embedder** | Consumes `frames.extracted`, runs CLIP ViT-B/32, stores vectors in pgVector, dedup by content hash |
| **Visualizer** | Streamlit app for exploring embeddings, similarity search, building and merging video flows |

## Visualizer Features

- **Embedding projection** — interactive scatter plot with PCA (instant, pre-computed), t-SNE, and UMAP
- **Similarity search** — click a point to find nearest neighbors; toggle cross-video mode to see one result per other video
- **Build Flow** — chain clips across videos using similarity; extract 5s clips and connect them into a sequence
- **Merge** — combine flow clips into a single video; concat (instant, no re-encode) or crossfade transitions (binary merge tree via xfade)
- **Collapsible UI** — sidebar controls organized into sections (Visualization, Search, Flow Settings, Advanced)

## Configuration

See `.env` for default settings. Key variables:

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_DSN` | `postgresql://iris:iris@postgres:5432/iris` | Database connection |
| `RABBITMQ_DSN` | `amqp://iris:iris@rabbitmq:5672/` | Message broker |
| `CACHE_DIR` | `/tmp/iris_cache` | Temporary clip and merge files |
| `FRAMES_DIR` | `/data/frames` | Frame JPEG storage |
| `DEBUG` | — | Set to `true` to enable verbose logging |

## GPU Acceleration

Uncomment the `deploy` block for the embedder service in `docker-compose.yml` to
enable CUDA acceleration for CLIP inference.

## Database Schema

- **videos** — id, filename, source_path, duration_s, status, content_hash, created_at
- **frames** — video_id, idx, timestamp_s, embedding (vector(512)), content_hash, pca_x, pca_y
- HNSW index on embedding for fast approximate nearest neighbor search
