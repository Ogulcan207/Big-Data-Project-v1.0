"""
Spark Structured Streaming Consumer
====================================

Kafka 'lending_club_stream' topic'ini dinler.
Veriyi Medallion Architecture (Bronze → Silver) ile Delta Lake'e yazar.

Mimari:
    Kafka Topic
         │
         ▼
    Spark Structured Streaming (readStream)
         │
         ▼
    JSON Parse + Schema Apply
         │
         ▼
    ┌─────────────────┐
    │  BRONZE (Ham)   │  ← Tüm kolonlar string, orijinal veri
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │  Temizleme +    │  ← Cast, parse, türetilmiş kolonlar
    │  Cast İşlemleri │
    └────────┬────────┘
             ▼
    ┌─────────────────┐
    │  SILVER (Temiz) │  ← Analize hazır, doğru tiplerde
    └─────────────────┘

Çalıştırma (Jupyter container içinde terminal):
    spark-submit consumer/spark_consumer.py

veya Python ile (paketler PYSPARK_SUBMIT_ARGS'tan otomatik gelir):
    python consumer/spark_consumer.py
"""

import sys
import os

# consumer/ klasörünü Python path'ine ekle (schema ve transformations import için)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp

from schema import LENDING_CLUB_SCHEMA
from transformations import bronze_to_silver


# ============================================================
# Konfigürasyon
# ============================================================
KAFKA_BOOTSTRAP_SERVERS = "kafka:29092"  # Container içi network adı
KAFKA_TOPIC = "lending_club_stream"

# Yollar (Jupyter container içinde /home/jovyan/work proje köküne mount)
BASE_PATH = "/home/jovyan/work"
BRONZE_PATH = f"{BASE_PATH}/lending_club_delta/bronze"
SILVER_PATH = f"{BASE_PATH}/lending_club_delta/silver"
BRONZE_CHECKPOINT = f"{BASE_PATH}/lending_club_checkpoint/bronze"
SILVER_CHECKPOINT = f"{BASE_PATH}/lending_club_checkpoint/silver"

# Streaming trigger süresi
TRIGGER_INTERVAL = "10 seconds"


# ============================================================
# 1. Spark Session
# ============================================================
def create_spark_session() -> SparkSession:
    """
    Delta Lake destekli Spark Session oluşturur.
    
    Kafka ve Delta paketleri docker-compose'daki PYSPARK_SUBMIT_ARGS
    ile otomatik geldiği için ekstra config'e gerek yok.
    """
    return (
        SparkSession.builder
        .appName("LendingClubStreamConsumer")
        # Delta Lake için zorunlu config'ler
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # Streaming için optimize edilmiş ayarlar
        .config("spark.sql.streaming.schemaInference", "true")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


# ============================================================
# 2. Kafka'dan Stream Okuma
# ============================================================
def read_from_kafka(spark: SparkSession):
    """
    Kafka topic'inden Structured Streaming ile okur.
    
    'startingOffsets=earliest' ile başlangıçta tüm topic'i okur.
    Production'da 'latest' kullanılır ama demo için tüm veriyi görmek isteriz.
    """
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")  # Eski offset silinmişse hata vermez
        .option("maxOffsetsPerTrigger", 10000)  # Her trigger'da max 10k mesaj
        .load()
    )


# ============================================================
# 3. JSON Parse → Bronze Şeması
# ============================================================
def parse_kafka_messages(kafka_df):
    """
    Kafka'dan gelen ham mesajları JSON parse eder ve şemaya uygular.
    
    Kafka mesajı binary olarak gelir, önce string'e cast edip
    sonra from_json ile structured hale getiririz.
    """
    return (
        kafka_df
        # Kafka metadata kolonları + ham value'yu string'e çevir
        .selectExpr(
            "CAST(key AS STRING) as kafka_key",
            "CAST(value AS STRING) as json_value",
            "topic as kafka_topic",
            "partition as kafka_partition",
            "offset as kafka_offset",
            "timestamp as kafka_timestamp"
        )
        # JSON'u şemaya göre parse et
        .withColumn("data", from_json(col("json_value"), LENDING_CLUB_SCHEMA))
        # Tüm kolonları top-level'a aç (data.* explode)
        .select(
            "kafka_topic", "kafka_partition", "kafka_offset",
            "kafka_timestamp", "data.*"
        )
        # Veri ne zaman ingest edildi? (audit)
        .withColumn("ingestion_timestamp", current_timestamp())
    )


# ============================================================
# 4. Bronze Katmana Yaz
# ============================================================
def write_bronze(parsed_df):
    """
    Bronze katman: Ham veri, hiçbir transformasyon yok.
    Veri kaynağının "as-is" kopyası. Delta formatında saklanır
    böylece ACID garantili, time travel yapılabilir.
    """
    return (
        parsed_df.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", BRONZE_CHECKPOINT)
        .option("path", BRONZE_PATH)
        .trigger(processingTime=TRIGGER_INTERVAL)
        .queryName("bronze_writer")
        .start()
    )


# ============================================================
# 5. Bronze'dan Silver'a Stream
# ============================================================
def stream_bronze_to_silver(spark: SparkSession):
    """
    Bronze Delta tablosunu kaynak olarak okur, transformasyon uygular,
    Silver Delta'ya yazar. Bu da ayrı bir streaming query.
    
    Delta Lake'in güçlü yanı: Bir Delta tablosu hem bir streaming query'nin
    sink'i hem de başka bir streaming query'nin source'u olabilir.
    """
    bronze_stream = (
        spark.readStream
        .format("delta")
        .load(BRONZE_PATH)
    )
    
    # Transformasyon pipeline'ı uygula
    silver_df = bronze_to_silver(bronze_stream)
    
    return (
        silver_df.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", SILVER_CHECKPOINT)
        .option("path", SILVER_PATH)
        .trigger(processingTime=TRIGGER_INTERVAL)
        .queryName("silver_writer")
        .start()
    )


# ============================================================
# 6. Konsolda İzleme (Opsiyonel - Debug)
# ============================================================
def write_to_console(parsed_df):
    """
    Debug için konsola da yazar. Production'da kapatılır.
    İlk akışı görmek için faydalı.
    """
    return (
        parsed_df.select("id", "loan_amnt", "grade", "loan_status",
                         "addr_state", "ingestion_timestamp")
        .writeStream
        .format("console")
        .outputMode("append")
        .option("truncate", "false")
        .option("numRows", 5)
        .trigger(processingTime="30 seconds")
        .queryName("console_monitor")
        .start()
    )


# ============================================================
# Main
# ============================================================
def wait_for_bronze_table(spark: SparkSession, max_wait_seconds: int = 300):
    """
    Silver writer başlamadan önce Bronze tablosunun oluşmasını bekler.
    
    Bronze tablosu, ilk batch yazıldığında oluşur. Eğer Silver bundan
    önce başlarsa DELTA_SCHEMA_NOT_SET hatası alır.
    
    Bu fonksiyon Bronze'da en az 1 satır olana kadar bekler.
    """
    import time
    from delta.tables import DeltaTable
    
    print("\n⏳ Bronze tablosunda ilk veri bekleniyor...")
    print("   (Producer çalışmıyorsa şimdi başlatın: yeni terminalde 'python producer.py')")
    
    waited = 0
    check_interval = 5  # Saniyede bir kontrol
    
    while waited < max_wait_seconds:
        try:
            if DeltaTable.isDeltaTable(spark, BRONZE_PATH):
                count = spark.read.format("delta").load(BRONZE_PATH).count()
                if count > 0:
                    print(f"✓ Bronze tablosunda {count} satır bulundu, Silver başlatılıyor.")
                    return True
        except Exception:
            pass  # Tablo henüz yok, beklemeye devam
        
        time.sleep(check_interval)
        waited += check_interval
        print(f"   ... bekleniyor ({waited}s / {max_wait_seconds}s)")
    
    print(f"⚠ {max_wait_seconds} saniye geçti, Bronze hâlâ boş. Silver yine de başlatılıyor.")
    return False


def main():
    print("=" * 60)
    print("Lending Club Spark Streaming Consumer Başlatılıyor")
    print("=" * 60)
    print(f"Kafka Broker  : {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Kafka Topic   : {KAFKA_TOPIC}")
    print(f"Bronze Path   : {BRONZE_PATH}")
    print(f"Silver Path   : {SILVER_PATH}")
    print(f"Trigger       : {TRIGGER_INTERVAL}")
    print("=" * 60)
    
    # 1. Spark session
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")  # INFO çok gürültülü
    
    # 2. Kafka'dan oku ve parse et
    kafka_df = read_from_kafka(spark)
    parsed_df = parse_kafka_messages(kafka_df)
    
    # 3. Bronze writer'ı başlat (Kafka → Bronze Delta)
    bronze_query = write_bronze(parsed_df)
    print(f"✓ Bronze writer başladı: {bronze_query.id}")
    
    # 4. Console monitor (debug)
    console_query = write_to_console(parsed_df)
    print(f"✓ Console monitor başladı: {console_query.id}")
    
    # 5. Silver writer için Bronze'un ilk batch'ini bekle
    #    (Silver, Bronze'u kaynak olarak okuduğu için tablo şeması gerekir)
    wait_for_bronze_table(spark)
    
    # 6. Silver writer'ı başlat (Bronze Delta → Silver Delta)
    silver_query = stream_bronze_to_silver(spark)
    print(f"✓ Silver writer başladı: {silver_query.id}")
    
    print("\n" + "=" * 60)
    print("Tüm streaming query'ler aktif. Ctrl+C ile durdur.")
    print("Spark UI: http://localhost:4040")
    print("=" * 60 + "\n")
    
    # 7. Tüm query'lerin sonsuza kadar çalışmasını bekle
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
