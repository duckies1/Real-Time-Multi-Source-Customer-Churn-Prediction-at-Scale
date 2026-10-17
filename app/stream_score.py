"""Score Kafka JSON feature events with the trained Spark ML pipeline; persist to HDFS and Kafka."""
import os
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml import PipelineModel

spark = SparkSession.builder.appName("kkbox-churn-kafka-score").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
model = PipelineModel.load(os.environ.get("MODEL_PATH", "hdfs://namenode:9000/churn/models/churn-rf"))
predictions_path = os.environ.get("PREDICTIONS_PATH", "hdfs://namenode:9000/churn/predictions")
checkpoint_path = os.environ.get("CHECKPOINT_PATH", "hdfs://namenode:9000/churn/checkpoints/score")
features_topic = os.environ.get("FEATURES_TOPIC", "churn-features")
predictions_topic = os.environ.get("PREDICTIONS_TOPIC", "churn-predictions")
numeric = ["city", "age", "registered_via", "tx_count", "avg_paid", "avg_plan_days", "auto_renew_rate", "ever_cancelled", "last_transaction_date", "log_days", "total_secs", "plays_25", "plays_50", "plays_75", "plays_985", "plays_100", "unique_songs"]
schema = T.StructType([T.StructField("msno", T.StringType())] + [T.StructField(c, T.DoubleType()) for c in numeric])
source = (spark.readStream.format("kafka").option("kafka.bootstrap.servers", "kafka:29092").option("subscribe", features_topic).option("startingOffsets", "earliest").load())
events = source.select(F.from_json(F.col("value").cast("string"), schema).alias("e")).select("e.*").where(F.col("msno").isNotNull())

def score_batch(batch, batch_id):
    if batch.rdd.isEmpty():
        return
    result = model.transform(batch).select("msno", F.col("prediction").cast("int").alias("prediction"), F.col("probability").cast("string").alias("probability"), F.lit(batch_id).alias("batch_id"))
    result.write.mode("append").parquet(predictions_path)
    (result.select(F.to_json(F.struct("msno", "prediction", "probability", "batch_id")).alias("value"))
       .write.format("kafka").option("kafka.bootstrap.servers", "kafka:29092").option("topic", predictions_topic).save())

query = (events.writeStream.foreachBatch(score_batch).option("checkpointLocation", checkpoint_path).trigger(processingTime="10 seconds").start())
query.awaitTermination()
