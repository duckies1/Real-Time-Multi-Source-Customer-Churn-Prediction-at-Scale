"""Small end-to-end smoke: feature two labeled users, fit MLlib, score one user."""
import os
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier

BASE = os.environ.get("DATA_DIR", "/dataset")
paths = {
    "train": os.environ.get("TRAIN_FILE", f"{BASE}/train_v2.csv/data/churn_comp_refresh/train_v2.csv"),
    "tx": os.environ.get("TRANSACTIONS_FILE", f"{BASE}/transactions_v2.csv/data/churn_comp_refresh/transactions_v2.csv"),
    "logs": os.environ.get("LOGS_FILE", f"{BASE}/user_logs_v2.csv/data/churn_comp_refresh/user_logs_v2.csv"),
    "members": os.environ.get("MEMBERS_FILE", f"{BASE}/members_v3.csv"),
}
def local_file(path):
    return path if path.startswith("file:") else f"file://{path}"
out = "hdfs://namenode:9000/churn/smoke-one"
spark = SparkSession.builder.appName("kkbox-smoke-one-example").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

labels = spark.read.option("header", True).csv(local_file(paths["train"])).select("msno", F.col("is_churn").cast("double").alias("label")).dropDuplicates(["msno"])
# A binary classifier cannot be fitted from one class. Use the selected churn
# example plus one non-churn control for the tiny fit; only the churn example
# is scored and persisted below.
positive = labels.where(F.col("label") == 1).select("msno", "label").limit(1)
negative = labels.where(F.col("label") == 0).select("msno", "label").limit(1)
tiny_labels = positive.unionByName(negative).cache()
selected = tiny_labels.collect()
if len(selected) != 2 or {r.label for r in selected} != {0.0, 1.0}:
    raise RuntimeError("Could not find one labeled user in each class")
target_id = next(r.msno for r in selected if r.label == 1.0)
ids = [r.msno for r in selected]

members = spark.read.option("header", True).csv(local_file(paths["members"])).where(F.col("msno").isin(ids)).select(
    "msno", F.col("city").cast("double").alias("city"), F.col("bd").cast("double").alias("age"), F.col("registered_via").cast("double").alias("registered_via"))
tx = spark.read.option("header", True).csv(local_file(paths["tx"])).where(F.col("msno").isin(ids))
txf = tx.groupBy("msno").agg(F.count("*").alias("tx_count"), F.avg(F.col("actual_amount_paid").cast("double")).alias("avg_paid"), F.avg(F.col("payment_plan_days").cast("double")).alias("avg_plan_days"), F.avg(F.col("is_auto_renew").cast("double")).alias("auto_renew_rate"), F.max(F.col("is_cancel").cast("double")).alias("ever_cancelled"), F.max(F.col("transaction_date").cast("int")).cast("double").alias("last_transaction_date"))
logs = spark.read.option("header", True).csv(local_file(paths["logs"])).where(F.col("msno").isin(ids))
logf = logs.groupBy("msno").agg(F.count("*").alias("log_days"), F.sum(F.col("total_secs").cast("double")).alias("total_secs"), F.sum(F.col("num_25").cast("double")).alias("plays_25"), F.sum(F.col("num_50").cast("double")).alias("plays_50"), F.sum(F.col("num_75").cast("double")).alias("plays_75"), F.sum(F.col("num_985").cast("double")).alias("plays_985"), F.sum(F.col("num_100").cast("double")).alias("plays_100"), F.sum(F.col("num_unq").cast("double")).alias("unique_songs"))
numeric = ["city", "age", "registered_via", "tx_count", "avg_paid", "avg_plan_days", "auto_renew_rate", "ever_cancelled", "last_transaction_date", "log_days", "total_secs", "plays_25", "plays_50", "plays_75", "plays_985", "plays_100", "unique_songs"]
rows = tiny_labels.join(members, "msno", "left").join(txf, "msno", "left").join(logf, "msno", "left").select("msno", "label", *numeric).fillna(0, subset=numeric).cache()
feature_count = rows.count()
if feature_count != 2:
    raise RuntimeError(f"Expected 2 users after multi-source joins, got {feature_count}")
imputed = [f"{c}_imp" for c in numeric]
pipeline = Pipeline(stages=[Imputer(inputCols=numeric, outputCols=imputed, strategy="median"), VectorAssembler(inputCols=imputed, outputCol="raw_features"), StandardScaler(inputCol="raw_features", outputCol="features"), RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=2, maxDepth=2, seed=27)])
model = pipeline.fit(rows)
scored = model.transform(rows.where(F.col("msno") == target_id)).select("msno", "label", "prediction", F.col("probability").cast("string").alias("probability"))
scored.write.mode("overwrite").json(out)
print(f"SMOKE_OK source_feature_users={feature_count} score_target={target_id}")
scored.show(truncate=False)
spark.stop()
