# KKBox real-time customer churn pipeline

An end-to-end, multi-source churn scoring demo using the WSDM KKBox dataset. It connects **Spark SQL / Spark MLlib**, **Kafka**, and **Hadoop HDFS**:

1. Spark joins the member, transaction, and listening-log tables to user-level features, evaluates and trains a Spark ML random forest, and stores features and the fitted model in HDFS.
2. A replay producer publishes the same joined feature rows to Kafka as JSON events (with a configurable event rate).
3. Spark Structured Streaming consumes those Kafka events, applies the persisted Spark ML pipeline, writes predictions to HDFS, and publishes prediction events to a second Kafka topic.

This is a practical replay-based streaming demo: the feature snapshot represents the competition's labeled training cohort, while Kafka supplies those records over time to the streaming scoring stage. It does not claim to reproduce a production online feature store or the competition's official leaderboard score.

## Prerequisites

- Docker Engine and Docker Compose v2
- Python 3.9+ on the host (for the Kafka replay producer)
- About 12 GB free disk for the v2 dataset plus Docker images and HDFS output. Original challenge logs are much larger; allow considerably more disk if using them.

## Dataset

The repository includes the KKBox dataset under `dataset/`. The default uses the compact refresh files already in this checkout:

```text
dataset/train_v2.csv/data/churn_comp_refresh/train_v2.csv
dataset/transactions_v2.csv/data/churn_comp_refresh/transactions_v2.csv
dataset/user_logs_v2.csv/data/churn_comp_refresh/user_logs_v2.csv
dataset/members_v3.csv
```

The code reads only the CSV data; it does not use Kaggle competition tooling or the old Scala labeler. If you need to obtain the source files separately, download them from Kaggle (account and competition access may be required):

```bash
python -m pip install kaggle
# Configure ~/.kaggle/kaggle.json using Kaggle's account instructions, then:
mkdir -p dataset/download
kaggle competitions download -c kkbox-churn-prediction-challenge -p dataset/download
unzip dataset/download/kkbox-churn-prediction-challenge.zip -d dataset/download
```

The challenge archive provides `train.csv`, `transactions.csv`, `user_logs.csv`, and `members_v3.csv`. To use those instead of the included refresh data, pass paths as environment variables on the Spark commands below, for example `TRAIN_FILE=/dataset/download/train.csv TRANSACTIONS_FILE=/dataset/download/transactions.csv LOGS_FILE=/dataset/download/user_logs.csv MEMBERS_FILE=/dataset/download/members_v3.csv`. Dataset paths inside the Spark container are under `/dataset` because Compose mounts `./dataset` there. If downloaded elsewhere, place the files under `dataset/download` first.

The default v2 files use a nested `data/churn_comp_refresh` path. If your copy is laid out differently, set the corresponding `*_FILE` environment variable.

## Run the pipeline

Run commands from the repository root. The first startup downloads the Docker images. The Spark image installs NumPy for Spark ML. Spark's Kafka connector is resolved by `/opt/spark/bin/spark-submit --packages` at job launch, so allow network access from Docker for Maven dependencies.

### 1. Start HDFS, Kafka, and the Spark runtime

```bash
mkdir -p artifacts
docker compose up -d --build
docker compose ps
```

HDFS NameNode UI: <http://localhost:9870>. Kafka is exposed at `localhost:9092`; the containers use `kafka:29092`.

### 2. Build source-joined features and train the model

Create the HDFS output directories, then run the Spark batch job:

```bash
docker compose exec namenode hdfs dfs -mkdir -p /churn/features /churn/models /churn/predictions /churn/checkpoints
docker compose exec spark /opt/spark/bin/spark-submit \
  --master 'local[*]' \
  --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
  /app/build_train.py
```

Successful completion prints a deterministic holdout `areaUnderROC` and saves the fitted pipeline model at `hdfs://namenode:9000/churn/models/churn-rf`. Feature rows are saved under `/churn/features/train`. The same features are exported into the host-mounted `artifacts/scoring-events/` folder for replay. The training step scans the listening history and can take several minutes.

For the downloaded original files, set paths for the Spark submit invocation:

```bash
docker compose exec -e TRAIN_FILE=/dataset/download/train.csv \
  -e TRANSACTIONS_FILE=/dataset/download/transactions.csv \
  -e LOGS_FILE=/dataset/download/user_logs.csv \
  -e MEMBERS_FILE=/dataset/download/members_v3.csv \
  spark /opt/spark/bin/spark-submit --master 'local[*]' \
  --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 /app/build_train.py
```

### Optional: one-target smoke run

This fast-path check takes one churn user as its scoring target and one non-churn control so the binary classifier sees both classes. It builds features from all three source tables, fits a tiny Spark ML random forest, scores the single target, and writes the result to `/churn/smoke-one` without touching the full model or feature outputs:

```bash
docker compose exec spark /opt/spark/bin/spark-submit \
  --master 'local[*]' \
  --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
  /app/smoke_one.py
docker compose exec namenode hdfs dfs -ls /churn/smoke-one
```

### 3. Start the streaming scorer

In a second terminal, start a long-running Structured Streaming job. The Spark connector is version-matched to Spark 3.5.1 / Scala 2.12:

```bash
docker compose exec spark /opt/spark/bin/spark-submit \
  --master 'local[*]' \
  --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \
  /app/stream_score.py
```

Leave it running. It reads the `churn-features` topic from its earliest offset, then writes append-only Parquet predictions to `/churn/predictions` and JSON prediction events to Kafka topic `churn-predictions`.

### 4. Replay features to Kafka

In a third terminal, install the small host-side producer dependency and send feature events. `--rate` is events per second; `--limit` is useful for a quick sample:

```bash
python -m pip install kafka-python==2.2.15
python app/produce_events.py --rate 100 --limit 10000
```

To replay the full cohort, omit `--limit`:

```bash
python app/produce_events.py --rate 100
```

The producer is safe to restart: the stream checkpoint preserves Kafka progress/offsets. For a clean repeat from the beginning, remove the checkpoint and prediction output first:

```bash
docker compose exec namenode hdfs dfs -rm -r -f /churn/checkpoints/score /churn/predictions
```

### 5. Inspect the outputs

```bash
docker compose exec namenode hdfs dfs -ls -R /churn
```

Read the Parquet output with Spark (the predictions are not plain text):

```bash
docker compose exec spark /opt/spark/bin/spark-sql \
  -e "SELECT msno, prediction, probability FROM parquet.`hdfs://namenode:9000/churn/predictions` LIMIT 20"
```

To inspect Kafka prediction messages, install `kcat` or use a Kafka UI/consumer and read topic `churn-predictions` from the beginning.

## Stop and cleanup

```bash
docker compose down        # stop containers and preserve HDFS volumes
docker compose down -v     # also delete HDFS volumes and all HDFS-stored model/output data
```

The HDFS volumes are persistent across `docker compose down`; host-side replay exports are in `artifacts/`.

## Feature and model notes

Features are aggregated per `msno`: member city/age/registration channel, transaction count/payment/plan/renewal/cancellation summaries, and listening days/seconds/play-count summaries. Missing numeric values are imputed, assembled and scaled by a Spark ML pipeline before random-forest classification. The label is the provided `is_churn` field. The pipeline is intentionally straightforward and reproducible; production use should add temporal split validation, leakage review, feature freshness policies, model monitoring, and secure Kafka/HDFS settings.

## Services and data path

```text
KKBox CSV sources --Spark SQL joins/aggregates--> HDFS features + Spark ML model
        HDFS-backed scoring feature export --> Kafka churn-features
        Kafka --> Spark Structured Streaming + loaded Spark ML model
             --> HDFS Parquet predictions
             --> Kafka churn-predictions
```
# Real-Time-Multi-Source-Customer-Churn-Prediction-at-Scale
