import json
import logging
import os
import signal
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Event

import cv2
import numpy as np
import pika
import psycopg

logger = logging.getLogger("iris-frame-extractor")

HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8080"))
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://iris:iris@postgres:5432/iris")
FRAMES_DIR = os.getenv("FRAMES_DIR", "/data/frames")

INPUT_QUEUE = "iris.video.new"
DLX_EXCHANGE = "iris.dlx"
INGEST_EXCHANGE = "iris.ingest"
PROCESSING_EXCHANGE = "iris.processing"
OUTPUT_ROUTING_KEY = "frames.extracted"

stopping = Event()


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            status_code = 200 if not stopping.is_set() else 503
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok" if not stopping.is_set() else "stopping"}).encode())
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
            channel.queue_declare(queue=INPUT_QUEUE, durable=True, arguments={
                "x-dead-letter-exchange": DLX_EXCHANGE,
                "x-message-ttl": 3600000,
            })
            channel.queue_bind(INPUT_QUEUE, INGEST_EXCHANGE, routing_key="video.new")
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
            logger.info("Connected to PostgreSQL")
            return conn
        except psycopg.OperationalError as e:
            logger.warning("PostgreSQL not ready, retrying in %ds: %s", delay, e)
            time.sleep(delay)
            delay = min(delay * 2, 30)


def is_video_pending(db_conn, video_id):
    with db_conn.cursor() as cur:
        cur.execute("SELECT status FROM videos WHERE id = %s", (video_id,))
        row = cur.fetchone()
        if row is None:
            return True
        return row[0] == "pending"


def set_video_processing(db_conn, video_id, content_hash, filename, source_path, size_bytes, duration_s, width, height, fps):
    with db_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO videos (id, filename, source_path, content_hash, size_bytes, duration_s, width, height, original_fps, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'processing')
            ON CONFLICT (id) DO UPDATE SET
                status = 'processing',
                updated_at = NOW()
            WHERE videos.status = 'pending'
        """, (video_id, filename, source_path, content_hash, size_bytes, duration_s, width, height, fps))
    db_conn.commit()


def set_video_error(db_conn, video_id, error_msg):
    with db_conn.cursor() as cur:
        cur.execute("""
            UPDATE videos SET status = 'error', error_msg = %s, updated_at = NOW()
            WHERE id = %s
        """, (error_msg, video_id))
    db_conn.commit()


def extract_frames(video_path, frames_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None, None, None, None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or total_frames <= 0:
        duration_s = 0
    else:
        duration_s = total_frames / fps

    frame_paths = []
    timestamps = []

    os.makedirs(frames_dir, exist_ok=True)

    for t in range(int(duration_s) + 1):
        target_frame = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        success, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            logger.warning("Failed to encode frame at t=%d", t)
            continue

        frame_path = os.path.join(frames_dir, f"{t:06d}.jpg")
        with open(frame_path, "wb") as f:
            f.write(buf)

        frame_paths.append(frame_path)
        timestamps.append(float(t))

    cap.release()
    return frame_paths, timestamps, duration_s, width, height, fps


def publish_frames(channel, video_id, filename, content_hash, frame_paths, timestamps, duration_s, width, height, fps):
    payload = {
        "video_id": video_id,
        "filename": filename,
        "content_hash": content_hash,
        "frame_paths": frame_paths,
        "timestamps": timestamps,
        "fps_original": fps,
        "duration_s": duration_s,
        "width": width,
        "height": height,
        "total_frames": len(frame_paths),
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    channel.basic_publish(
        exchange=PROCESSING_EXCHANGE,
        routing_key=OUTPUT_ROUTING_KEY,
        body=json.dumps(payload).encode(),
        properties=pika.BasicProperties(delivery_mode=2),
    )
    logger.info("Published frames.extracted: video_id=%s frames=%d",
                 video_id, len(frame_paths))


def nack_to_dlx(channel, delivery_tag, reason):
    logger.error("NACK to DLX: %s", reason)
    channel.basic_nack(delivery_tag=delivery_tag, requeue=False)


def callback(ch, method, properties, body, db_conn, publish_channel):
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
    filename = msg["filename"]
    source_path = msg["path"]
    content_hash = msg.get("content_hash", "")
    size_bytes = msg.get("size_bytes", 0)

    logger.info("Processing video.new: video_id=%s filename=%s hash=%s", video_id, filename, content_hash)

    if not is_video_pending(db_conn, video_id):
        logger.info("Video %s already processed, skipping", video_id)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    frames_dir = os.path.join(FRAMES_DIR, video_id)
    result = extract_frames(source_path, frames_dir)
    if result[0] is None:
        set_video_error(db_conn, video_id, "Could not open video file")
        nack_to_dlx(ch, method.delivery_tag, f"Could not open video: {source_path}")
        return

    frame_paths, timestamps, duration_s, width, height, fps = result

    if not frame_paths:
        set_video_error(db_conn, video_id, "No frames extracted")
        nack_to_dlx(ch, method.delivery_tag, f"No frames extracted: {source_path}")
        return

    set_video_processing(db_conn, video_id, content_hash, filename, source_path, size_bytes, duration_s, width, height, fps)

    try:
        publish_frames(publish_channel, video_id, filename, content_hash, frame_paths, timestamps, duration_s, width, height, fps)
    except Exception as e:
        set_video_error(db_conn, video_id, f"Publish failed: {e}")
        logger.error("Publish failed for %s, message will be requeued: %s", video_id, e)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        return

    ch.basic_ack(delivery_tag=method.delivery_tag)
    logger.info("Done: video_id=%s frames=%d duration=%.1fs", video_id, len(frame_paths), duration_s)


def main():
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    )

    os.makedirs(FRAMES_DIR, exist_ok=True)

    health_server = start_health_server()

    def shutdown(*args):
        logger.info("Shutting down...")
        stopping.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    db_conn = connect_db()
    connection, channel = connect_rabbitmq()

    channel.basic_qos(prefetch_count=1)

    def on_message(ch, method, properties, body):
        callback(ch, method, properties, body, db_conn, channel)

    channel.basic_consume(queue=INPUT_QUEUE, on_message_callback=on_message, auto_ack=False)

    logger.info("Waiting for messages on %s...", INPUT_QUEUE)
    try:
        channel.start_consuming()
    except Exception:
        pass
    finally:
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
