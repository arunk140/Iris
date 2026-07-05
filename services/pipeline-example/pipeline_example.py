"""
pipeline-example — Template for adding new pipeline stages.

This service demonstrates the pattern for extending Project Iris:
1. Choose an exchange and routing key to consume from.
2. Process the message (replace the log statement with real logic).
3. Publish results to a new routing key for downstream consumers.

To wire it into the pipeline, uncomment the `pipeline-example` service
in docker-compose.yml and configure the source exchange/routing key.
"""

import json
import logging
import os
import signal
import time

import pika

logger = logging.getLogger("iris-pipeline-example")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_VHOST = os.getenv("RABBITMQ_VHOST", "/")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")

# --- CONFIGURE YOUR PIPELINE STAGE HERE ---
# Source: where to consume messages from
SOURCE_EXCHANGE = os.getenv("EXAMPLE_SOURCE_EXCHANGE", "iris.ingest")
SOURCE_ROUTING_KEY = os.getenv("EXAMPLE_SOURCE_ROUTING_KEY", "video.new")
SOURCE_QUEUE = os.getenv("EXAMPLE_SOURCE_QUEUE", "iris.example.video")

# Destination: where to publish results (empty = no publish)
DEST_EXCHANGE = os.getenv("EXAMPLE_DEST_EXCHANGE", "iris.example")
DEST_ROUTING_KEY = os.getenv("EXAMPLE_DEST_ROUTING_KEY", "example.processed")

stopping = False


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
    while not stopping:
        try:
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.exchange_declare(exchange=SOURCE_EXCHANGE, exchange_type="direct", durable=True)
            if DEST_EXCHANGE:
                channel.exchange_declare(exchange=DEST_EXCHANGE, exchange_type="direct", durable=True)
            channel.queue_declare(queue=SOURCE_QUEUE, durable=True)
            channel.queue_bind(SOURCE_QUEUE, SOURCE_EXCHANGE, routing_key=SOURCE_ROUTING_KEY)
            logger.info(
                "Connected: consuming from %s / %s → %s",
                SOURCE_EXCHANGE, SOURCE_ROUTING_KEY, SOURCE_QUEUE,
            )
            return connection, channel
        except pika.exceptions.AMQPConnectionError as e:
            logger.warning("RabbitMQ not ready, retrying in %ds: %s", delay, e)
            time.sleep(delay)
            delay = min(delay * 2, 30)


def callback(ch, method, properties, body):
    if stopping:
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        return

    try:
        msg = json.loads(body)
    except json.JSONDecodeError:
        msg = body

    # --- YOUR PIPELINE LOGIC GOES HERE ---
    # Replace this log with your actual processing (e.g., ML inference, DB write, etc.)
    logger.info(
        "Received message from %s: routing_key=%s delivery_tag=%s",
        SOURCE_EXCHANGE, method.routing_key, method.delivery_tag,
    )
    logger.info("Message body: %s", json.dumps(msg, indent=2)[:500])
    # --- END PIPELINE LOGIC ---

    # Optionally publish results to a downstream exchange
    if DEST_EXCHANGE:
        result = {
            "original_message": msg,
            "processed": True,
            "example_note": "Replace this with your output payload",
        }
        ch.basic_publish(
            exchange=DEST_EXCHANGE,
            routing_key=DEST_ROUTING_KEY,
            body=json.dumps(result),
            properties=pika.BasicProperties(delivery_mode=2),
        )
        logger.info("Published to %s / %s", DEST_EXCHANGE, DEST_ROUTING_KEY)

    ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper()),
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    )

    global stopping

    def shutdown(*args):
        logger.info("Shutting down...")
        stopping = True

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    connection, channel = connect_rabbitmq()
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=SOURCE_QUEUE, on_message_callback=callback, auto_ack=False)

    logger.info("Waiting for messages on %s...", SOURCE_QUEUE)
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
        logger.info("Stopped")


if __name__ == "__main__":
    main()
