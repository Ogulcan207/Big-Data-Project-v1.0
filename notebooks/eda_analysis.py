from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("Gold EDA Analysis")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    return spark


if __name__ == "__main__":
    spark = create_spark_session()

    gold_path = "/home/jovyan/work/lending_club_delta/gold"

    print("[1/6] Gold katmanı okunuyor...")
    gold_df = spark.read.format("delta").load(gold_path)

    print("Gold kayıt sayısı:", gold_df.count())
    print("Gold kolon sayısı:", len(gold_df.columns))

    print("\n[2/6] Şema bilgisi:")
    gold_df.printSchema()

    print("\n[3/6] Label dağılımı:")
    gold_df.groupBy("label").count().orderBy("label").show()

    print("\n[4/6] Loan status dağılımı:")
    gold_df.groupBy("loan_status").count().orderBy(desc("count")).show(20, truncate=False)

    print("\n[5/6] Sayısal kolonlar için özet istatistikler:")

    numeric_columns = [
        "loan_amnt",
        "funded_amnt",
        "int_rate",
        "installment",
        "annual_inc",
        "dti",
        "fico_avg",
        "loan_to_income_ratio",
        "revol_bal",
        "total_pymnt",
        "recoveries"
    ]

    existing_numeric_columns = [
        c for c in numeric_columns if c in gold_df.columns
    ]

    gold_df.select(existing_numeric_columns).describe().show(truncate=False)

    print("\n[6/6] Kategorik kolon dağılımları:")

    categorical_columns = [
        "grade",
        "sub_grade",
        "home_ownership",
        "verification_status",
        "purpose",
        "application_type",
        "term"
    ]

    for column_name in categorical_columns:
        if column_name in gold_df.columns:
            print(f"\n{column_name} dağılımı:")
            gold_df.groupBy(column_name).count().orderBy(desc("count")).show(20, truncate=False)

    print("\nÖrnek veri:")
    gold_df.select(
        "loan_amnt",
        "int_rate",
        "annual_inc",
        "dti",
        "grade",
        "purpose",
        "loan_status",
        "label"
    ).show(10, truncate=False)

    print("\nEDA analizi tamamlandı.")

    spark.stop()