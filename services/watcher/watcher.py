import glob
import hashlib
import json
import logging
import os
import re
import signal
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Event

import pika

logger = logging.getLogger("iris-watcher")

HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8080"))
SOURCE_DIR = os.getenv("SOURCE_DIR", "/data/source")
WATCH_EXT = os.getenv("WATCH_EXT", ".mp4,.mov")
WATCH_EXTS = [ext.strip() for ext in WATCH_EXT.split(",")]
POLL_INTERVAL_S = int(os.getenv("POLL_INTERVAL_S", "10"))
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")

EXCHANGE = "iris.ingest"
ROUTING_KEY = "video.new"

stopping = Event()


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            healthy = not stopping.is_set()
            status_code = 200 if healthy else 503
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok" if healthy else "stopping"}).encode())
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
            channel.exchange_declare(exchange=EXCHANGE, exchange_type="direct", durable=True)
            logger.info("Connected to RabbitMQ")
            return connection, channel
        except pika.exceptions.AMQPConnectionError as e:
            logger.warning("RabbitMQ not ready, retrying in %ds: %s", delay, e)
            time.sleep(delay)
            delay = min(delay * 2, 30)


def is_file_stable(path, check_interval_s=1):
    try:
        s1 = os.stat(path)
        time.sleep(check_interval_s)
        s2 = os.stat(path)
        return s1.st_size == s2.st_size and s1.st_mtime == s2.st_mtime
    except FileNotFoundError:
        return False


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def is_uuid_name(filepath):
    name, _ = os.path.splitext(os.path.basename(filepath))
    return bool(UUID_PATTERN.match(name))


def poll_source():
    while not stopping.is_set():
        conn_channel = connect_rabbitmq()
        if conn_channel is None:
            break
        connection, channel = conn_channel
        try:
            while not stopping.is_set():
                try:
                    files = set()
                    for ext in WATCH_EXTS:
                        pattern = os.path.join(SOURCE_DIR, f"*{ext}")
                        files.update(glob.glob(pattern))
                    files = sorted(files)
                    for filepath in files:
                        if stopping.is_set():
                            return
                        if is_uuid_name(filepath):
                            continue
                        if not is_file_stable(filepath):
                            logger.info("File not stable, skipping: %s", filepath)
                            continue
                        if not process_file(channel, filepath):
                            raise ConnectionError("Publish failed, reconnecting")
                except (pika.exceptions.AMQPError, ConnectionError) as e:
                    logger.error("Connection error, reconnecting: %s", e)
                    break
                except Exception as e:
                    logger.error("Poll iteration failed: %s", e, exc_info=True)

                stopping.wait(POLL_INTERVAL_S)
        finally:
            try:
                connection.close()
            except Exception:
                pass


def process_file(channel, filepath):
    filename = os.path.basename(filepath)
    _, ext = os.path.splitext(filename)
    video_id = str(uuid.uuid4())
    new_name = f"{video_id}{ext}"
    new_path = os.path.join(SOURCE_DIR, new_name)

    try:
        os.rename(filepath, new_path)
    except OSError as e:
        logger.error("Failed to rename %s -> %s: %s", filepath, new_path, e)
        return False

    size_bytes = os.path.getsize(new_path)

    logger.info("Computing SHA-256 for %s...", new_path)
    content_hash = hash_file(new_path)
    logger.info("SHA-256: %s", content_hash)

    message = {
        "video_id": video_id,
        "filename": filename,
        "path": new_path,
        "content_hash": content_hash,
        "size_bytes": size_bytes,
        "detected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    try:
        channel.basic_publish(
            exchange=EXCHANGE,
            routing_key=ROUTING_KEY,
            body=json.dumps(message),
            properties=pika.BasicProperties(delivery_mode=2),
        )
        logger.info(
            "Published video.new: video_id=%s filename=%s size=%d hash=%s",
            video_id, filename, size_bytes, content_hash,
        )
        return True
    except Exception as e:
        logger.error("Publish failed for %s, renaming back: %s", video_id, e)
        try:
            os.rename(new_path, filepath)
        except OSError:
            logger.error("Could not rename back %s -> %s", new_path, filepath)
        return False


def main():
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    )

    os.makedirs(SOURCE_DIR, exist_ok=True)
    logger.info("Watching directory: %s (exts=%s, interval=%ds)", SOURCE_DIR, WATCH_EXTS, POLL_INTERVAL_S)

    health_server = start_health_server()

    def shutdown(*args):
        logger.info("Shutting down...")
        stopping.set()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    poll_source()


if __name__ == "__main__":
    main()
