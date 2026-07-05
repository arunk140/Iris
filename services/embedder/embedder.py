import json
import logging
import os
import signal
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Event, Lock

import cv2
import numpy as np
from PIL import Image
import pika
import psycopg
import torch
from pgvector.psycopg import register_vector

logger = logging.getLogger("iris-embedder")

HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8080"))
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://iris:iris@postgres:5432/iris")
FRAMES_DIR = os.getenv("FRAMES_DIR", "/data/frames")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "ViT-B-32")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
BATCH_TTL_S = int(os.getenv("BATCH_TTL_S", "300"))

INPUT_QUEUE = "iris.frames.extracted"
DLX_EXCHANGE = "iris.dlx"
INGEST_EXCHANGE = "iris.ingest"
PROCESSING_EXCHANGE = "iris.processing"

device = "cuda" if torch.cuda.is_available() else "cpu"
stopping = Event()

# Accumulator for split batches: video_id -> dict
accumulator = {}
accum_lock = Lock()
model = None
preprocess = None


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            issues = []
            if stopping.is_set():
                issues.append("stopping")
            if model is None:
                issues.append("model not loaded")
            status_code = 200 if not issues else 503
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "ok" if not issues else "error",
                "issues": issues,
            }).encode())
        elif self.path == "/ready":
            ready = model is not None and not stopping.is_set()
            self.send_response(200 if ready else 503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ready": ready}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        logger.debug("HTTP: %s", format % args)


def start_health_server():
    server = HTTPServer(("0.0.0.0", HEALTH_PORT), HealthHandler)
    t = Thread(target=server.serve_forever, daemon=True)
    t.start()
    logger.info("Health server listening on port %d", HEALTH_PORT)
    return server


def connect_rabbitmq():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    params = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        virtual_host=RABBITMQ_VHOST,
        credentials=credentials,
        heartbeat=60,
        blocked_connection_timeout=30,
    )
    delay = 1
    while not stopping.is_set():
        try:
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.exchange_declare(exchange=INGEST_EXCHANGE, exchange_type="direct", durable=True)
            channel.exchange_declare(exchange=PROCESSING_EXCHANGE, exchange_type="direct", durable=True)
            channel.exchange_declare(exchange=DLX_EXCHANGE, exchange_type="fanout", durable=True)
            channel.queue_declare(queue=INPUT_QUEUE, durable=True, arguments={
                "x-dead-letter-exchange": DLX_EXCHANGE,
                "x-message-ttl": 3600000,
            })
            channel.queue_declare(queue="iris.dlq", durable=True)
            channel.queue_bind(INPUT_QUEUE, PROCESSING_EXCHANGE, routing_key="frames.extracted")
            channel.queue_bind("iris.dlq", DLX_EXCHANGE)
            logger.info("Connected to RabbitMQ")
            return connection, channel
        except pika.exceptions.AMQPConnectionError as e:
            logger.warning("RabbitMQ not ready, retrying in %ds: %s", delay, e)
            time.sleep(delay)
            delay = min(delay * 2, 30)


def connect_db():
    delay = 1
    while not stopping.is_set():
        try:
            conn = psycopg.connect(POSTGRES_DSN)
            register_vector(conn)
            logger.info("Connected to PostgreSQL")
            return conn
        except psycopg.OperationalError as e:
            logger.warning("PostgreSQL not ready, retrying in %ds: %s", delay, e)
            time.sleep(delay)
            delay = min(delay * 2, 30)


def run_migrations(db_conn):
    candidates = [
        "/app/db/init.sql",
        "/app/init.sql",
        os.path.join(os.path.dirname(__file__), "..", "..", "db", "init.sql"),
    ]
    path = None
    for c in candidates:
        if os.path.exists(c):
            path = c
            break
    if path:
        with open(path) as f:
            sql = f.read()
        with db_conn.cursor() as cur:
            cur.execute(sql)
        db_conn.commit()
        logger.info("Schema migrations applied from %s", path)
    else:
        logger.warning("No init.sql found (searched %s)", candidates)


def load_model():
    global model, preprocess
    logger.info("Loading CLIP model %s on %s...", EMBEDDING_MODEL, device)
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        EMBEDDING_MODEL, pretrained="openai", device=device,
    )
    model.eval()
    logger.info("Model loaded")


def load_frame_from_disk(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def embed_frames(frames_np):
    global model, preprocess
    all_embeddings = []

    for i in range(0, len(frames_np), EMBEDDING_BATCH_SIZE):
        batch = frames_np[i:i + EMBEDDING_BATCH_SIZE]
        tensors = torch.stack([preprocess(Image.fromarray(f)) for f in batch]).to(device)
        with torch.no_grad():
            emb = model.encode_image(tensors)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings)


def store_embeddings(db_conn, video_id, content_hash, filename, source_path, size_bytes, duration_s,
                     width, height, fps, timestamps, embeddings):
    with db_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO videos (id, filename, source_path, content_hash, size_bytes, duration_s,
                                width, height, original_fps, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'done')
            ON CONFLICT (id) DO UPDATE SET
                content_hash = EXCLUDED.content_hash,
                status = 'done', updated_at = NOW()
            WHERE videos.status = 'processing' OR videos.status = 'pending'
        """, (video_id, filename, source_path, content_hash, size_bytes, duration_s, width, height, fps))

        rows = []
        for idx, (ts, emb) in enumerate(zip(timestamps, embeddings)):
            rows.append((video_id, idx, ts, emb))

        cur.executemany(
            "INSERT INTO frames (video_id, idx, timestamp_s, embedding) "
            "VALUES (%s, %s, %s, %s) ON CONFLICT (video_id, idx) DO NOTHING",
            rows,
        )
    db_conn.commit()


def process_batch(db_conn, video_id, content_hash, filename, source_path, size_bytes, duration_s,
                  width, height, fps, frame_paths, timestamps):
    logger.info("Embedding %d frames for video_id=%s", len(frame_paths), video_id)

    frames_np = []
    valid_timestamps = []
    for path, ts in zip(frame_paths, timestamps):
        img = load_frame_from_disk(path)
        if img is None:
            logger.warning("Skipping corrupt frame at t=%.1f for video_id=%s: %s", ts, video_id, path)
            continue
        frames_np.append(img)
        valid_timestamps.append(ts)

    if not frames_np:
        logger.error("No valid frames to embed for video_id=%s", video_id)
        with db_conn.cursor() as cur:
            cur.execute("UPDATE videos SET status = 'error', error_msg = 'No valid frames', updated_at = NOW() WHERE id = %s", (video_id,))
        db_conn.commit()
        return

    embeddings = embed_frames(frames_np)
    if embeddings.shape[0] != len(valid_timestamps):
        logger.warning("Embedding count mismatch: %d vs %d timestamps", embeddings.shape[0], len(valid_timestamps))

    store_embeddings(
        db_conn, video_id, content_hash, filename, source_path, size_bytes, duration_s,
        width, height, fps, valid_timestamps, embeddings,
    )
    logger.info(
        "Stored %d embeddings for video_id=%s (duration=%.1fs)",
        len(valid_timestamps), video_id, duration_s,
    )


def flush_accumulated(video_id, db_conn):
    with accum_lock:
        acc = accumulator.pop(video_id, None)
    if acc is None:
        return

    sorted_batches = sorted(acc["batches"], key=lambda x: x[0])
    frame_paths = []
    timestamps = []
    for _, fp, ts in sorted_batches:
        frame_paths.extend(fp)
        timestamps.extend(ts)

    process_batch(
        db_conn, video_id, acc["content_hash"], acc["filename"], acc["source_path"], acc["size_bytes"],
        acc["duration_s"], acc["width"], acc["height"], acc["fps"],
        frame_paths, timestamps,
    )

def cleanup_expired(db_conn):
    now = time.monotonic()
    expired = []
    with accum_lock:
        for vid, acc in list(accumulator.items()):
            if now > acc["deadline"]:
                expired.append(vid)
    for vid in expired:
        logger.warning("Batch TTL expired for video_id=%s, processing partial data", vid)
        flush_accumulated(vid, db_conn)


def callback(ch, method, properties, body, db_conn):
    if stopping.is_set():
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        return

    try:
        msg = json.loads(body)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON message: %s", e)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        return

    video_id = msg["video_id"]
    filename = msg.get("filename", "")
    source_path = ""
    size_bytes = 0
    content_hash = msg.get("content_hash", "")
    duration_s = msg.get("duration_s", 0)
    width = msg.get("width", 0)
    height = msg.get("height", 0)
    fps = msg.get("fps_original", 0)
    frame_paths = msg.get("frame_paths", [])
    timestamps = msg.get("timestamps", [])
    batch_index = msg.get("batch_index")
    total_batches = msg.get("total_batches", 1)

    # Deduplication check: skip if content_hash already processed
    if content_hash:
        with db_conn.cursor() as cur:
            cur.execute("SELECT id FROM videos WHERE content_hash = %s AND status = 'done'", (content_hash,))
            existing = cur.fetchone()
            if existing:
                logger.info("Duplicate content_hash=%s (existing video_id=%s), skipping video_id=%s",
                            content_hash, existing[0], video_id)
                with db_conn.cursor() as cur:
                    cur.execute("UPDATE videos SET status = 'done', updated_at = NOW() WHERE id = %s", (video_id,))
                db_conn.commit()
                ch.basic_ack(delivery_tag=method.delivery_tag)
                return

    # Idempotency check
    with db_conn.cursor() as cur:
        cur.execute("SELECT status FROM videos WHERE id = %s", (video_id,))
        row = cur.fetchone()
        if row and row[0] == "done":
            logger.info("Video %s already done, skipping", video_id)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

    if total_batches == 1 and batch_index is None:
        try:
            process_batch(
                db_conn, video_id, content_hash, filename, source_path, size_bytes, duration_s,
                width, height, fps, frame_paths, timestamps,
            )
        except Exception as e:
            logger.error("Failed to process batch for video_id=%s: %s", video_id, e, exc_info=True)
            with db_conn.cursor() as cur:
                cur.execute("UPDATE videos SET status = 'error', error_msg = %s, updated_at = NOW() WHERE id = %s",
                            (str(e), video_id))
                db_conn.commit()
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    with accum_lock:
        if video_id not in accumulator:
            accumulator[video_id] = {
                "content_hash": content_hash,
                "filename": filename,
                "source_path": source_path,
                "size_bytes": size_bytes,
                "duration_s": duration_s,
                "width": width,
                "height": height,
                "fps": fps,
                "batches": [],
                "deadline": time.monotonic() + BATCH_TTL_S,
                "total_batches": total_batches,
            }
        acc = accumulator[video_id]
        acc["batches"].append((batch_index, frame_paths, timestamps))

        if len(acc["batches"]) >= acc["total_batches"]:
            logger.info("All %d batches received for video_id=%s", total_batches, video_id)
            ch.basic_ack(delivery_tag=method.delivery_tag)
            flush_accumulated(video_id, db_conn)
            return

    ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    )
    logger.info("Using device: %s", device)

    health_server = start_health_server()

    def shutdown(*args):
        logger.info("Shutting down...")
        stopping.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    db_conn = connect_db()
    run_migrations(db_conn)

    load_model()

    connection, channel = connect_rabbitmq()
    channel.basic_qos(prefetch_count=1)

    def on_message(ch, method, properties, body):
        callback(ch, method, properties, body, db_conn)

    channel.basic_consume(queue=INPUT_QUEUE, on_message_callback=on_message, auto_ack=False)

    # Background thread to clean up expired batch accumulators
    def cleanup_loop():
        while not stopping.is_set():
            time.sleep(30)
            try:
                cleanup_expired(db_conn)
            except Exception as e:
                logger.error("Cleanup error: %s", e)

    Thread(target=cleanup_loop, daemon=True).start()

    logger.info("Waiting for messages on %s...", INPUT_QUEUE)
    try:
        channel.start_consuming()
    except Exception as e:
        logger.error("Consumer exited unexpectedly: %s", e, exc_info=True)
    finally:
        # Flush any remaining accumulated batches
        with accum_lock:
            remaining = list(accumulator.keys())
        for vid in remaining:
            flush_accumulated(vid, db_conn)
        try:
            channel.stop_consuming()
            connection.close()
        except Exception:
            pass
        db_conn.close()
        health_server.shutdown()
        logger.info("Stopped")


if __name__ == "__main__":
    main()
