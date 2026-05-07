"""
Bronze → Silver Dönüşüm Fonksiyonları
======================================

Bronze katman: Tüm kolonlar string, ham veri (Kafka'dan geldiği gibi).
Silver katman: Doğru tipler, temizlenmiş, analize hazır.

Yapılan dönüşümler:
1. Yüzde kolonlarındaki "%" işaretini temizle (örn. "13.99%" → 13.99)
2. Sayısal kolonları DoubleType'a cast et
3. Tarih kolonlarını DateType'a parse et (Lending Club formatı: "Dec-2015")
4. Boş string'leri NULL'a çevir
5. Türetilmiş analitik kolonlar ekle (loan_to_income_ratio, fico_avg vb.)
6. Audit kolonları ekle (ingestion_timestamp, processing_timestamp)
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, when, regexp_replace, trim, to_date,
    current_timestamp, lit
)
from pyspark.sql.types import DoubleType

from schema import NUMERIC_COLUMNS, DATE_COLUMNS, PERCENT_COLUMNS


def clean_empty_strings(df: DataFrame) -> DataFrame:
    """
    Boş string ve sadece whitespace içeren değerleri NULL'a çevirir.
    Lending Club CSV'sinde "" yerine bazen " " gelebiliyor.
    """
    for column in df.columns:
        df = df.withColumn(
            column,
            when(trim(col(column)) == "", None).otherwise(col(column))
        )
    return df


def parse_percent_columns(df: DataFrame) -> DataFrame:
    """
    Yüzde kolonlarındaki "%" işaretini kaldırır ve double'a çevirir.
    Örnek: "13.99%" → 13.99
    """
    for column in PERCENT_COLUMNS:
        if column in df.columns:
            df = df.withColumn(
                column,
                regexp_replace(col(column), "%", "").cast(DoubleType())
            )
    return df


def cast_numeric_columns(df: DataFrame) -> DataFrame:
    """Sayısal kolonları DoubleType'a cast eder."""
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(DoubleType()))
    return df


def parse_date_columns(df: DataFrame) -> DataFrame:
    """
    Tarih kolonlarını DateType'a parse eder.
    Lending Club formatı: "Dec-2015" → 2015-12-01
    """
    for column in DATE_COLUMNS:
        if column in df.columns:
            df = df.withColumn(
                column,
                to_date(col(column), "MMM-yyyy")
            )
    return df


def add_derived_columns(df: DataFrame) -> DataFrame:
    """
    Analiz için türetilmiş kolonlar ekler.
    
    - fico_avg: FICO range'in ortalaması
    - loan_to_income_ratio: Kredi tutarı / yıllık gelir
    - is_high_risk: Yüksek riskli kredi flag'i (grade D, E, F, G)
    - is_charged_off: Temerrüt flag'i
    """
    # FICO ortalaması (kredi skoru)
    df = df.withColumn(
        "fico_avg",
        (col("fico_range_low") + col("fico_range_high")) / 2
    )
    
    # Loan-to-income oranı (büyük veri analitiğinde temel metrik)
    df = df.withColumn(
        "loan_to_income_ratio",
        when(col("annual_inc") > 0, col("loan_amnt") / col("annual_inc"))
        .otherwise(None)
    )
    
    # Yüksek risk flag'i
    df = df.withColumn(
        "is_high_risk",
        when(col("grade").isin("D", "E", "F", "G"), lit(True))
        .otherwise(lit(False))
    )
    
    # Temerrüt flag'i
    df = df.withColumn(
        "is_charged_off",
        when(col("loan_status").contains("Charged Off"), lit(True))
        .otherwise(lit(False))
    )
    
    return df


def add_audit_columns(df: DataFrame) -> DataFrame:
    """
    Veri lineage takibi için audit kolonları.
    Her satırın ne zaman işlendiği bilinir, debug için kritik.
    """
    return df.withColumn("processing_timestamp", current_timestamp())


def bronze_to_silver(df: DataFrame) -> DataFrame:
    """
    Bronze → Silver tam dönüşüm pipeline'ı.
    
    Sıralama önemli:
    1. Önce boş string'leri temizle (cast'ler null'lara takılmasın)
    2. Yüzdeleri parse et (regex)
    3. Sayısalları cast et
    4. Tarihleri parse et
    5. Türetilmiş kolonlar
    6. Audit kolonları
    """
    return (
        df
        .transform(clean_empty_strings)
        .transform(parse_percent_columns)
        .transform(cast_numeric_columns)
        .transform(parse_date_columns)
        .transform(add_derived_columns)
        .transform(add_audit_columns)
    )