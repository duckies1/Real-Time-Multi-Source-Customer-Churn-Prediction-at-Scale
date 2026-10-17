"""Small REST interface for the trained 100 MiB KKBox demo model."""
import json
import os
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

MODEL_PATH = os.getenv("MODEL_PATH", "hdfs://namenode:9000/churn/slice-100mb/models/churn-rf")
FEATURES_PATH = os.getenv("FEATURES_PATH", "hdfs://namenode:9000/churn/slice-100mb/features/train")
BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
PREDICTIONS_TOPIC = os.getenv("KAFKA_PREDICTIONS_TOPIC", "churn-ui-predictions")

spark = (SparkSession.builder.appName("kkbox-ui-inference")
         .master(os.getenv("SPARK_MASTER", "local[2]"))
         .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
         .config("spark.driver.memory", "2g")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")
model = PipelineModel.load(MODEL_PATH)
features = spark.read.parquet(FEATURES_PATH).cache()
feature_columns = [c for c in features.columns if c not in ("msno", "label")]
sample_rows = features.count()

app = FastAPI(title="KKBox Churn Inference API", version="1.0.0",
              description="Demo inference backed by the 100 MiB KKBox slice, HDFS and Spark ML.")
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins or ["*"], allow_methods=["*"],
                   allow_headers=["*"], allow_credentials=False)


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse("/app/ui/index.html")

try:
    from kafka import KafkaProducer
    producer = KafkaProducer(bootstrap_servers=BOOTSTRAP,
                             value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                             request_timeout_ms=3000, api_version_auto_timeout_ms=3000)
except Exception as exc:
    producer = None
    print("Kafka audit publishing unavailable:", exc, flush=True)


class PredictRequest(BaseModel):
    msno: Optional[str] = None
    features: Optional[Dict[str, float]] = None


def probability_list(vector):
    return [float(x) for x in vector.toArray().tolist()]


def score(frame):
    rows = model.transform(frame).select("msno", "prediction", "probability").collect()
    results = [{"msno": r.msno, "prediction": int(r.prediction),
                "churn_probability": probability_list(r.probability)[1]} for r in rows]
    if producer and results:
        for result in results:
            producer.send(PREDICTIONS_TOPIC, result)
        producer.flush(timeout=5)
    return results


def public_row(row):
    result = {"msno": row["msno"]}
    result.update({c: (float(row[c]) if row[c] is not None else 0.0) for c in feature_columns})
    if "label" in row.asDict():
        result["actual_label"] = int(row["label"])
    return result


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "feature_rows": sample_rows,
            "kafka_audit_enabled": producer is not None, "predictions_topic": PREDICTIONS_TOPIC}


@app.get("/api/v1/sample-users")
def sample_users(limit: int = Query(10, ge=1, le=100)):
    rows = features.orderBy("msno").limit(limit).collect()
    return {"count": len(rows), "users": [public_row(r) for r in rows]}


@app.post("/api/v1/predict")
def predict(request: PredictRequest):
    if bool(request.msno) == bool(request.features):
        raise HTTPException(422, "Provide exactly one of 'msno' or 'features'.")
    if request.msno:
        found = features.filter(features.msno == request.msno).limit(1).collect()
        if not found:
            raise HTTPException(404, "msno not found in the 100 MiB demo feature set")
        row = found[0]
        frame = spark.createDataFrame([row])
    else:
        missing = sorted(set(feature_columns) - set(request.features or {}))
        extra = sorted(set(request.features or {}) - set(feature_columns))
        if missing or extra:
            raise HTTPException(422, {"missing_features": missing, "unknown_features": extra})
        schema = StructType([StructField("msno", StringType(), False)] +
                            [StructField(c, DoubleType(), True) for c in feature_columns])
        values = [request.msno or "ui-custom-input"] + [float(request.features[c]) for c in feature_columns]
        frame = spark.createDataFrame([tuple(values)], schema)
    result = score(frame)[0]
    return {"model": "100mb-slice-random-forest", **result}


@app.post("/api/v1/simulate")
def simulate(limit: int = Query(20, ge=1, le=500)):
    rows = features.orderBy("msno").limit(limit).collect()
    if not rows:
        return {"count": 0, "predictions": [], "accuracy": None}
    scored = score(spark.createDataFrame(rows))
    labels = {r["msno"]: int(r["label"]) for r in rows}
    for item in scored:
        item["actual_label"] = labels.get(item["msno"])
    correct = sum(1 for item in scored if item["prediction"] == item["actual_label"])
    return {"count": len(scored), "accuracy": correct / len(scored), "predictions": scored}
