"""Build user-level features from KKBox's three source tables and train Spark ML."""
import os
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

BASE = os.environ.get("DATA_DIR", "/dataset")
TRAIN = os.environ.get("TRAIN_FILE", f"{BASE}/train_v2.csv/data/churn_comp_refresh/train_v2.csv")
TX = os.environ.get("TRANSACTIONS_FILE", f"{BASE}/transactions_v2.csv/data/churn_comp_refresh/transactions_v2.csv")
LOG = os.environ.get("LOGS_FILE", f"{BASE}/user_logs_v2.csv/data/churn_comp_refresh/user_logs_v2.csv")
MEM = os.environ.get("MEMBERS_FILE", f"{BASE}/members_v3.csv")
def local_file(path):
    return path if path.startswith("file:") else f"file://{path}"
OUT = "hdfs://namenode:9000/churn"

spark = SparkSession.builder.appName("kkbox-churn-feature-train").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
labels = spark.read.option("header", True).csv(local_file(TRAIN)).select("msno", F.col("is_churn").cast("double").alias("label")).dropDuplicates(["msno"])
members = spark.read.option("header", True).csv(local_file(MEM)).select("msno", F.col("city").cast("double"), F.col("bd").cast("double").alias("age"), F.col("registered_via").cast("double"))
tx = spark.read.option("header", True).csv(local_file(TX))
tx_features = tx.groupBy("msno").agg(
    F.count("*").alias("tx_count"), F.avg(F.col("actual_amount_paid").cast("double")).alias("avg_paid"),
    F.avg(F.col("payment_plan_days").cast("double")).alias("avg_plan_days"),
    F.avg(F.col("is_auto_renew").cast("double")).alias("auto_renew_rate"),
    F.max(F.col("is_cancel").cast("double")).alias("ever_cancelled"),
    F.max(F.col("transaction_date").cast("int")).alias("last_transaction_date"))
logs = spark.read.option("header", True).csv(local_file(LOG))
log_features = logs.groupBy("msno").agg(
    F.count("*").alias("log_days"), F.sum(F.col("total_secs").cast("double")).alias("total_secs"),
    F.sum(F.col("num_25").cast("double")).alias("plays_25"), F.sum(F.col("num_50").cast("double")).alias("plays_50"),
    F.sum(F.col("num_75").cast("double")).alias("plays_75"), F.sum(F.col("num_985").cast("double")).alias("plays_985"),
    F.sum(F.col("num_100").cast("double")).alias("plays_100"), F.sum(F.col("num_unq").cast("double")).alias("unique_songs"))
features = labels.join(members, "msno", "left").join(tx_features, "msno", "left").join(log_features, "msno", "left")
numeric = ["city", "age", "registered_via", "tx_count", "avg_paid", "avg_plan_days", "auto_renew_rate", "ever_cancelled", "last_transaction_date", "log_days", "total_secs", "plays_25", "plays_50", "plays_75", "plays_985", "plays_100", "unique_songs"]
features = features.select("msno", "label", *numeric).fillna(0, subset=numeric)
features.write.mode("overwrite").parquet(f"{OUT}/features/train")
# deterministic holdout provides a useful metric without leaking records across user rows
train_df, test_df = features.randomSplit([0.8, 0.2], seed=27)
imputed = [f"{c}_imp" for c in numeric]
pipeline = Pipeline(stages=[Imputer(inputCols=numeric, outputCols=imputed, strategy="median"), VectorAssembler(inputCols=imputed, outputCol="raw_features"), StandardScaler(inputCol="raw_features", outputCol="features"), RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=80, maxDepth=8, seed=27)])
model = pipeline.fit(train_df)
metric = BinaryClassificationEvaluator(labelCol="label").evaluate(model.transform(test_df))
print(f"Holdout areaUnderROC={metric:.5f}")
model.write().overwrite().save(f"{OUT}/models/churn-rf")
# Small Kafka replay file, generated using the exact same feature transformations.
scoring = features.drop("label")
scoring.coalesce(1).write.mode("overwrite").json("file:///artifacts/scoring-events")
spark.stop()
