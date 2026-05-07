"""
CSV → Bronze → Silver Batch Backfill
=====================================

Tüm CSV'yi (1.6 GB, ~2.2M satır) tek seferlik batch olarak işler.
Streaming pipeline ile aynı dönüşüm mantığını kullanır (transformations.py).

Bu script şunu garantiler:
- Bronze ve Silver şemaları streaming ile birebir aynı
- Aynı 'bronze_to_silver()' fonksiyonu kullanılır → tutarlılık
- Idempotent: overwrite mode ile her çalıştırmada aynı sonuç
"""

import sys
from pathlib import Path

# consumer/ klasörünü Python path'ine ekle (schema.py ve transformations.py için)
sys.path.insert(0, str(Path(__file__).parent))

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType

from schema import LENDING_CLUB_SCHEMA  # 151 kolonun StringType şeması
from transformations import bronze_to_silver


# ===== Yapılandırma =====
CSV_PATH = "/home/jovyan/work/accepted_2007_to_2018Q4.csv"
BRONZE_PATH = "/home/jovyan/work/lending_club_delta/bronze"
SILVER_PATH = "/home/jovyan/work/lending_club_delta/silver"


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("LendingClub-Backfill-CSV-to-Silver")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def add_kafka_metadata_placeholders(df):
    """
    Streaming pipeline'da Kafka'dan gelen metadata kolonlarını ekler.
    Backfill'de Kafka kullanılmadığı için NULL/sabit değerlerle doldurulur.
    Bu, Bronze şemasının streaming ile birebir aynı kalmasını sağlar.
    """
    return (
        df
        .withColumn("kafka_topic", lit("backfill_csv"))
        .withColumn("kafka_partition", lit(None).cast("int"))
        .withColumn("kafka_offset", lit(None).cast("long"))
        .withColumn("kafka_timestamp", lit(None).cast("timestamp"))
        .withColumn("ingestion_timestamp", current_timestamp())
    )


def main():
    print("=" * 70)
    print("LENDING CLUB BACKFILL: CSV → Bronze → Silver")
    print("=" * 70)
    
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    # ===== ADIM 1: CSV'yi tüm kolonları string olarak oku =====
    print("\n[1/5] CSV okunuyor (tüm kolonlar StringType)...")
    df_raw = (
        spark.read
        .option("header", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .option("quote", '"')
        .schema(LENDING_CLUB_SCHEMA)
        .csv(CSV_PATH)
    )
    raw_count = df_raw.count()
    print(f"      → CSV'den okunan satır sayısı: {raw_count:,}")
    print(f"      → Kolon sayısı: {len(df_raw.columns)}")
    
    # ===== ADIM 2: Kafka metadata placeholder'ları ekle =====
    print("\n[2/5] Kafka metadata kolonları ekleniyor (placeholder)...")
    df_bronze = add_kafka_metadata_placeholders(df_raw)
    print(f"      → Bronze kolon sayısı: {len(df_bronze.columns)}")
    
    # ===== ADIM 3: Bronze'a yaz =====
    print(f"\n[3/5] Bronze'a yazılıyor → {BRONZE_PATH}")
    (
        df_bronze.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(BRONZE_PATH)
    )
    print("      → Bronze yazıldı ✓")
    
    # ===== ADIM 4: Bronze → Silver dönüşümü =====
    print("\n[4/5] Silver dönüşümü uygulanıyor (transformations.bronze_to_silver)...")
    df_bronze_read = spark.read.format("delta").load(BRONZE_PATH)
    df_silver = bronze_to_silver(df_bronze_read)
    print(f"      → Silver kolon sayısı: {len(df_silver.columns)}")
    
    # ===== ADIM 5: Silver'a yaz =====
    print(f"\n[5/5] Silver'a yazılıyor → {SILVER_PATH}")
    (
        df_silver.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(SILVER_PATH)
    )
    print("      → Silver yazıldı ✓")
    
    # ===== Doğrulama =====
    print("\n" + "=" * 70)
    print("DOĞRULAMA")
    print("=" * 70)
    
    bronze_count = spark.read.format("delta").load(BRONZE_PATH).count()
    silver_count = spark.read.format("delta").load(SILVER_PATH).count()
    
    print(f"Bronze kayıt sayısı:  {bronze_count:,}")
    print(f"Silver kayıt sayısı:  {silver_count:,}")
    print(f"CSV ham kayıt sayısı: {raw_count:,}")
    
    if bronze_count == silver_count == raw_count:
        print("\n✓ Tüm sayılar tutarlı. Backfill başarılı.")
    else:
        print("\n⚠ Sayı uyumsuzluğu var, kontrol et.")
    
    spark.stop()


if __name__ == "__main__":
    main()
