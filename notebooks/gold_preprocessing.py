from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count
from pyspark.sql.types import NumericType, StringType


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("Gold Preprocessing")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    return spark


if __name__ == "__main__":
    spark = create_spark_session()

    silver_path = "/home/jovyan/work/lending_club_delta/silver"
    gold_path = "/home/jovyan/work/lending_club_delta/gold"

    print("[1/6] Silver katmanı okunuyor...")
    silver_df = spark.read.format("delta").load(silver_path)

    print("Silver kayıt sayısı:", silver_df.count())
    print("Silver kolon sayısı:", len(silver_df.columns))

    print("[2/6] Label oluşturuluyor...")

    gold_df = silver_df.withColumn(
        "label",
        when(
            col("loan_status").isin(
                "Charged Off",
                "Default",
                "Does not meet the credit policy. Status:Charged Off"
            ),
            1
        ).otherwise(0)
    )

    print("[3/6] Gereksiz ve çok eksik kolonlar siliniyor...")

    drop_columns = [
        "id",
        "member_id",
        "url",
        "desc",

        "mths_since_last_record",
        "mths_since_last_major_derog",
        "mths_since_recent_bc_dlq",
        "mths_since_recent_revol_delinq",

        "annual_inc_joint",
        "dti_joint",
        "verification_status_joint",

        "revol_bal_joint",
        "sec_app_fico_range_low",
        "sec_app_fico_range_high",
        "sec_app_earliest_cr_line",
        "sec_app_inq_last_6mths",
        "sec_app_mort_acc",
        "sec_app_open_acc",
        "sec_app_revol_util",
        "sec_app_open_act_il",
        "sec_app_num_rev_accts",
        "sec_app_chargeoff_within_12_mths",
        "sec_app_collections_12_mths_ex_med",
        "sec_app_mths_since_last_major_derog",

        "hardship_type",
        "hardship_reason",
        "hardship_status",
        "deferral_term",
        "hardship_amount",
        "hardship_start_date",
        "hardship_end_date",
        "payment_plan_start_date",
        "hardship_length",
        "hardship_dpd",
        "hardship_loan_status",
        "orig_projected_additional_accrued_interest",
        "hardship_payoff_balance_amount",
        "hardship_last_payment_amount",

        "debt_settlement_flag_date",
        "settlement_status",
        "settlement_date",
        "settlement_amount",
        "settlement_percentage",
        "settlement_term",

        "kafka_partition",
        "kafka_offset",
        "kafka_timestamp"
    ]

    existing_drop_columns = [c for c in drop_columns if c in gold_df.columns]
    gold_df = gold_df.drop(*existing_drop_columns)

    print("Silinen kolon sayısı:", len(existing_drop_columns))
    print("Kolon silme sonrası kolon sayısı:", len(gold_df.columns))

    print("[4/6] Eksik değerler dolduruluyor...")

    numeric_cols = [
        field.name
        for field in gold_df.schema.fields
        if isinstance(field.dataType, NumericType)
    ]

    string_cols = [
        field.name
        for field in gold_df.schema.fields
        if isinstance(field.dataType, StringType)
    ]

    gold_df = gold_df.fillna(0, subset=numeric_cols)
    gold_df = gold_df.fillna("Unknown", subset=string_cols)

    print("Sayısal kolon sayısı:", len(numeric_cols))
    print("Kategorik kolon sayısı:", len(string_cols))

    print("[5/6] Kontrol çıktıları alınıyor...")

    print("Label dağılımı:")
    gold_df.groupBy("label").count().show()

    print("Gold örnek veri:")
    gold_df.select("loan_amnt", "int_rate", "annual_inc", "loan_status", "label").show(10, truncate=False)

    print("[6/6] Gold katmanına yazılıyor...")

    (
        gold_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(gold_path)
    )

    print("Gold yazıldı.")
    print("Gold kayıt sayısı:", gold_df.count())
    print("Gold kolon sayısı:", len(gold_df.columns))

    spark.stop()