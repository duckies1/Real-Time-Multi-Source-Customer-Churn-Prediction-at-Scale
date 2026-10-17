"""Replay Spark-exported source-joined feature rows into Kafka at a controllable rate."""
import argparse, glob, json, os, time
from kafka import KafkaProducer
from kafka.errors import NoBrokersAvailable

p = argparse.ArgumentParser()
p.add_argument("--input", default="/artifacts/scoring-events")
p.add_argument("--rate", type=float, default=100.0, help="feature events per second")
p.add_argument("--limit", type=int, default=0, help="0 sends all exported users")
a = p.parse_args()
topic = os.environ.get("KAFKA_TOPIC", "churn-features")
files = glob.glob(os.path.join(a.input, "*.json"))
if not files:
    raise SystemExit(f"No exported JSON files at {a.input}; run build_train.py first")
for attempt in range(30):
    try:
        producer = KafkaProducer(bootstrap_servers=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"), value_serializer=lambda x: json.dumps(x).encode("utf-8"))
        break
    except NoBrokersAvailable:
        if attempt == 29: raise
        time.sleep(2)
sent = 0
delay = 1.0 / max(a.rate, 0.1)
for path in files:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if a.limit and sent >= a.limit: break
            producer.send(topic, json.loads(line))
            sent += 1
            if sent % 1000 == 0: producer.flush(); print(f"sent {sent} events")
            time.sleep(delay)
    if a.limit and sent >= a.limit: break
producer.flush(); producer.close()
print(f"sent {sent} feature events")
