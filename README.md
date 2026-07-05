# Project Iris — Video Intelligence Platform

Phase 1: Automatically ingest raw video files, extract one frame per second, embed
each frame with CLIP ViT-B/32 (512-dim), and store everything in PostgreSQL + pgVector.

```
source/foo.mp4 ──► Watcher ──► RabbitMQ ──► Frame Extractor ──► RabbitMQ ──► Embedder ──► PostgreSQL
```

## Quickstart

```bash
# Start everything
docker compose up --build -d

# Drop a video into source/
cp ~/clips/ride_001.mp4 source/

# Watch the pipeline
docker compose logs -f

# Query results
docker compose exec postgres psql -U iris -d iris -c "
  SELECT id, filename, duration_s, status FROM videos;
  SELECT video_id, COUNT(*) AS frames FROM frames GROUP BY video_id;
"
```

## Services

| Service | Role |
|---|---|
| **Watcher** | Polls `source/` for `.mp4` files, assigns UUID, renames, publishes to `iris.ingest` |
| **Frame Extractor** | Consumes `video.new`, extracts 1 frame/sec via OpenCV, publishes to `iris.processing` |
| **Embedder** | Consumes `frames.extracted`, runs CLIP ViT-B/32, stores vectors in pgVector |
| **pipeline-example** | Reference template for adding new stages (logs messages, see `Design.md`) |

## Debug Mode

```bash
DEBUG=true docker compose up -d
# Frames written to ./frames/{video_id}/
```

## GPU

Uncomment the `deploy` block for the embedder service in `docker-compose.yml` to
enable CUDA acceleration.

## Extending the Pipeline

Copy `services/pipeline-example/`, configure the source exchange and routing key
via env vars, replace the log statement with your logic, and add the service to
`docker-compose.yml`. See [`Design.md`](Design.md#hello-world-pipeline-example)
for details.

## Design

See [`Design.md`](Design.md) for the full architecture spec.
