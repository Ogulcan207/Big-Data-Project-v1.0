"""
Spark Structured Streaming Consumer (Canli Terminal Versiyonu)
==============================================================

Kafka 'lending_club_stream' topic'ini dinler.
Veriyi Medallion Architecture (Bronze -> Silver) ile Delta Lake'e yazar.

Bu versiyonda her batch sonrasi canli terminal ciktisi:
  - Kac mesaj geldi, kac toplam
  - Anlik hiz (msg/sn)
  - Bronze/Silver satir sayilari
  - Veri ornegi (3 satir)

Calistirma:
    docker exec -it spark-jupyter spark-submit \\
        --packages io.delta:delta-spark_2.12:3.0.0,org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \\
        /home/jovyan/work/consumer/spark_consumer.py
"""

import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp

from schema import LENDING_CLUB_SCHEMA
from transformations import bronze_to_silver


# ============================================================
# Konfigurasyon
# ============================================================
KAFKA_BOOTSTRAP_SERVERS = "kafka:29092"
KAFKA_TOPIC = "lending_club_stream"

BASE_PATH = "/home/jovyan/work"
BRONZE_PATH = f"{BASE_PATH}/lending_club_delta/bronze"
SILVER_PATH = f"{BASE_PATH}/lending_club_delta/silver"
BRONZE_CHECKPOINT = f"{BASE_PATH}/lending_club_checkpoint/bronze"
SILVER_CHECKPOINT = f"{BASE_PATH}/lending_club_checkpoint/silver"

TRIGGER_INTERVAL = "5 seconds"


# ============================================================
# Renkli yardimcilar
# ============================================================
class C:
    G = '\033[92m'
    Y = '\033[93m'
    B = '\033[94m'
    M = '\033[95m'
    C = '\033[96m'
    R = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def banner(text, color="\033[96m"):
    line = "=" * 70
    print(f"\n{color}{line}{C.END}")
    print(f"{color}{C.BOLD}  {text}{C.END}")
    print(f"{color}{line}{C.END}")


def section(text, color="\033[94m"):
    print(f"\n{color}{C.BOLD}>>> {text}{C.END}")
    print(f"{color}{'-' * 70}{C.END}")


def info(label, value, color="\033[92m"):
    print(f"   {color}{label:<35}{C.END} {C.BOLD}{value}{C.END}")


# ============================================================
# Spark Session
# ============================================================
def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("LendingClubStreamConsumer")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.streaming.schemaInference", "true")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


# ============================================================
# Kafka okuma + JSON parse
# ============================================================
def read_from_kafka(spark):
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .option("maxOffsetsPerTrigger", 10000)
        .load()
    )


def parse_kafka_messages(kafka_df):
    return (
        kafka_df
        .selectExpr(
            "CAST(key AS STRING) as kafka_key",
            "CAST(value AS STRING) as json_value",
            "topic as kafka_topic",
            "partition as kafka_partition",
            "offset as kafka_offset",
            "timestamp as kafka_timestamp"
        )
        .withColumn("data", from_json(col("json_value"), LENDING_CLUB_SCHEMA))
        .select(
            "kafka_topic", "kafka_partition", "kafka_offset",
            "kafka_timestamp", "data.*"
        )
        .withColumn("ingestion_timestamp", current_timestamp())
    )


# ============================================================
# Global state - batch sayaclari
# ============================================================
_state = {
    "bronze_total": 0,
    "bronze_batch_count": 0,
    "silver_total": 0,
    "silver_batch_count": 0,
    "start_time": time.time(),
    "last_batch_time": time.time(),
    "silver_last_time": time.time(),
}


# ============================================================
# Bronze foreachBatch -> canli print + Delta yazma
# ============================================================
def bronze_foreach_batch(batch_df, batch_id):
    batch_count = batch_df.count()
    _state["bronze_total"] += batch_count
    _state["bronze_batch_count"] += 1

    now = time.time()
    elapsed = now - _state["last_batch_time"]
    rate = batch_count / elapsed if elapsed > 0 else 0
    _state["last_batch_time"] = now

    total_elapsed = now - _state["start_time"]
    avg_rate = _state["bronze_total"] / total_elapsed if total_elapsed > 0 else 0

    banner(f"BRONZE BATCH #{batch_id}  |  {datetime.now().strftime('%H:%M:%S')}", C.M)
    info("Bu batch'te gelen mesaj sayisi", f"{batch_count:,}")
    info("Bu batch hizi (msg/sn)", f"{rate:,.1f}")
    info("Bronze toplam satir", f"{_state['bronze_total']:,}", C.Y)
    info("Toplam islem suresi", f"{total_elapsed:.1f} sn")
    info("Ortalama hiz (msg/sn)", f"{avg_rate:,.1f}", C.Y)

    if batch_count > 0:
        # Delta'ya yaz
        (batch_df.write
            .format("delta")
            .mode("append")
            .save(BRONZE_PATH))
        info("Bronze Delta'ya yazildi", "OK", C.G)

        # Ornek 3 satir
        section("Bronze'a yazilan canli ornek (3 satir):", C.G)
        try:
            sample_cols = [c for c in
                            ["id", "loan_amnt", "grade", "loan_status",
                             "addr_state", "ingestion_timestamp"]
                            if c in batch_df.columns]
            sample = batch_df.select(*sample_cols).limit(3).collect()
            for i, r in enumerate(sample, 1):
                vals = "  ".join(f"{c}={r[c]}" for c in sample_cols)
                print(f"   {C.G}[{i}]{C.END} {vals}")
        except Exception as e:
            print(f"   (ornek alinamadi: {e})")
    else:
        print(f"\n   {C.Y}Bu batch'te yeni mesaj yok, bekleniyor...{C.END}")


# ============================================================
# Silver foreachBatch -> canli print + Delta yazma
# ============================================================
def silver_foreach_batch(batch_df, batch_id):
    batch_count = batch_df.count()
    _state["silver_total"] += batch_count
    _state["silver_batch_count"] += 1

    now = time.time()
    elapsed = now - _state["silver_last_time"]
    rate = batch_count / elapsed if elapsed > 0 else 0
    _state["silver_last_time"] = now

    banner(f"SILVER BATCH #{batch_id}  |  {datetime.now().strftime('%H:%M:%S')}", C.C)
    info("Bu batch'te islenen satir", f"{batch_count:,}")
    info("Bu batch hizi (satir/sn)", f"{rate:,.1f}")
    info("Silver toplam satir", f"{_state['silver_total']:,}", C.Y)
    info("Bronze -> Silver donusum", "Tip cast + temizleme uygulandi", C.Y)
    info("Silver kolon sayisi", f"{len(batch_df.columns)}")

    if batch_count > 0:
        (batch_df.write
            .format("delta")
            .mode("append")
            .save(SILVER_PATH))
        info("Silver Delta'ya yazildi", "OK", C.G)

        section("Silver'a yazilan canli ornek (3 satir):", C.G)
        try:
            preferred = ["id", "loan_amnt", "int_rate", "grade",
                         "loan_status", "annual_inc", "default_flag"]
            silver_cols = [c for c in preferred if c in batch_df.columns]
            if not silver_cols:
                silver_cols = batch_df.columns[:6]
            sample = batch_df.select(*silver_cols).limit(3).collect()
            for i, r in enumerate(sample, 1):
                vals = "  ".join(f"{c}={r[c]}" for c in silver_cols)
                print(f"   {C.G}[{i}]{C.END} {vals}")
        except Exception as e:
            print(f"   (ornek alinamadi: {e})")
    else:
        print(f"\n   {C.Y}Silver tarafinda yeni satir yok.{C.END}")


# ============================================================
# Bronze writer (foreachBatch ile)
# ============================================================
def write_bronze(parsed_df):
    return (
        parsed_df.writeStream
        .foreachBatch(bronze_foreach_batch)
        .outputMode("append")
        .option("checkpointLocation", BRONZE_CHECKPOINT)
        .trigger(processingTime=TRIGGER_INTERVAL)
        .queryName("bronze_writer")
        .start()
    )


# ============================================================
# Silver writer (Bronze Delta -> Silver Delta)
# ============================================================
def stream_bronze_to_silver(spark):
    bronze_stream = (
        spark.readStream
        .format("delta")
        .load(BRONZE_PATH)
    )
    silver_df = bronze_to_silver(bronze_stream)

    return (
        silver_df.writeStream
        .foreachBatch(silver_foreach_batch)
        .outputMode("append")
        .option("checkpointLocation", SILVER_CHECKPOINT)
        .trigger(processingTime=TRIGGER_INTERVAL)
        .queryName("silver_writer")
        .start()
    )


# ============================================================
# Bronze tablonun olusmasini bekle (silver baslamadan once)
# ============================================================
def wait_for_bronze_table(spark, max_wait_seconds=300):
    from delta.tables import DeltaTable

    banner("BRONZE TABLOSU BEKLENIYOR", C.Y)
    print(f"   {C.Y}Producer calismiyorsa simdi baslatin:{C.END}")
    print(f"   {C.C}  python producer.py  (yeni terminalde){C.END}\n")

    waited = 0
    check_interval = 5

    while waited < max_wait_seconds:
        try:
            if DeltaTable.isDeltaTable(spark, BRONZE_PATH):
                count_n = spark.read.format("delta").load(BRONZE_PATH).count()
                if count_n > 0:
                    print(f"{C.G}{C.BOLD}   --> Bronze hazir: {count_n:,} satir bulundu, Silver baslatiliyor.{C.END}")
                    return True
        except Exception:
            pass

        time.sleep(check_interval)
        waited += check_interval
        # Donen progress bar
        dots = "." * ((waited // check_interval) % 4)
        print(f"   {C.Y}Bekleniyor{dots:<3} ({waited}s / {max_wait_seconds}s){C.END}")

    print(f"{C.R}   {max_wait_seconds} saniye gecti, Bronze hala bos. Silver yine de baslatiliyor.{C.END}")
    return False


# ============================================================
# Main
# ============================================================
def main():
    banner("LENDING CLUB SPARK STREAMING CONSUMER", C.C)
    info("Kafka Broker", KAFKA_BOOTSTRAP_SERVERS)
    info("Kafka Topic", KAFKA_TOPIC)
    info("Bronze Path", BRONZE_PATH)
    info("Silver Path", SILVER_PATH)
    info("Trigger Interval", TRIGGER_INTERVAL)
    info("Baslangic Zamani", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    section("1. Kafka stream'i acaliyor...", C.B)
    kafka_df = read_from_kafka(spark)
    parsed_df = parse_kafka_messages(kafka_df)
    print(f"   {C.G}OK{C.END} Kafka readStream + JSON parse hazir")

    section("2. Bronze writer baslatiliyor (Kafka -> Bronze Delta)...", C.B)
    bronze_query = write_bronze(parsed_df)
    print(f"   {C.G}OK{C.END} Bronze writer ID: {bronze_query.id}")

    wait_for_bronze_table(spark)

    section("3. Silver writer baslatiliyor (Bronze Delta -> Silver Delta)...", C.B)
    silver_query = stream_bronze_to_silver(spark)
    print(f"   {C.G}OK{C.END} Silver writer ID: {silver_query.id}")

    banner("TUM STREAMING QUERY'LER AKTIF", C.G)
    info("Bronze Query", "RUNNING", C.G)
    info("Silver Query", "RUNNING", C.G)
    info("Spark UI", "http://localhost:4040")
    print(f"\n   {C.Y}Ctrl+C ile durdurabilirsiniz.{C.END}")
    print(f"   {C.Y}Her {TRIGGER_INTERVAL} arayla yeni batch raporu gelecek...{C.END}\n")

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()