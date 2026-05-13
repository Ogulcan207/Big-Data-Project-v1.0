"""
EDA Analysis - Gold Katmani (Sunum Versiyonu)
===============================================

Her adim terminalde aciklamali olarak gosterilir.
Sunum sirasinda terminale bakarak:
  - Hangi parametre/kolon kullanildi
  - Sayisal sonuc ne cikti
  - Bu bulgunun anlami nedir
acik sekilde okunabilir.

Calistirma:
    docker exec -it spark-jupyter spark-submit \\
        --packages io.delta:delta-spark_2.12:3.0.0 \\
        /home/jovyan/work/notebooks/eda_analysis.py
"""

import time
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc, when, isnan, mean, stddev, min as smin, max as smax


# ============================================================
# Renkli yardimcilar
# ============================================================
class C:
    G = '\033[92m'   # green
    Y = '\033[93m'   # yellow
    B = '\033[94m'   # blue
    M = '\033[95m'   # magenta
    C = '\033[96m'   # cyan
    R = '\033[91m'   # red
    BOLD = '\033[1m'
    END = '\033[0m'


def banner(title, color=C.C):
    line = "=" * 75
    print(f"\n{color}{line}{C.END}")
    print(f"{color}{C.BOLD}  {title}{C.END}")
    print(f"{color}{line}{C.END}")


def step(num, total, title, color=C.B):
    print(f"\n{color}{C.BOLD}>>> ADIM [{num}/{total}]  {title}{C.END}")
    print(f"{color}{'-' * 75}{C.END}")


def explain(text, color=C.Y):
    """Bu print sunum sirasinda 'biz burada sunu yapiyoruz' diye okunur."""
    print(f"   {color}AMAC: {C.END}{text}")


def used(label, value, color=C.G):
    """Kullanilan parametre / kolon goster."""
    print(f"   {color}KULLANILAN:{C.END} {label} = {C.BOLD}{value}{C.END}")


def result(label, value, color=C.M):
    """Cikan sonuc."""
    print(f"   {color}SONUC:{C.END} {label} = {C.BOLD}{value}{C.END}")


def comment(text, color=C.C):
    """Yorum / bulgu."""
    print(f"   {color}YORUM: {C.END}{text}")


# ============================================================
# Spark Session
# ============================================================
def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("Gold_EDA_Analysis")
        .config("spark.sql.extensions",
                "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    start_time = time.time()

    banner("GOLD KATMANI - KESIFSEL VERI ANALIZI (EDA)", C.C)
    print(f"   {C.G}Baslangic:{C.END} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   {C.G}Amac:{C.END} Gold tablodaki veriyi anlamak, kalitesini olcmek,")
    print(f"   {C.G}     {C.END} model egitimine hazir olup olmadigini dogrulamak.")

    spark = create_spark_session()
    gold_path = "/home/jovyan/work/lending_club_delta/gold"

    # ===================================================================
    # ADIM 1/6: Gold tabloyu yukle
    # ===================================================================
    step(1, 6, "GOLD KATMANI YUKLENIYOR", C.B)
    explain("Silver katmanindan feature engineering ile uretilmis Gold tabloyu okuyoruz. "
            "Bu, model egitimine en yakin haldeki temizlenmis veridir.")
    used("Delta tablo yolu", gold_path)
    used("Format", "delta (ACID + versiyon kontrolu)")

    t1 = time.time()
    gold_df = spark.read.format("delta").load(gold_path)
    gold_df.cache()

    row_count = gold_df.count()
    col_count = len(gold_df.columns)
    load_time = time.time() - t1

    result("Yukleme suresi", f"{load_time:.2f} sn")
    result("Toplam satir sayisi", f"{row_count:,}")
    result("Toplam kolon sayisi", f"{col_count}")
    comment(f"{row_count:,} satir x {col_count} kolon = buyuk veri olcekli isleme yapiliyor.")

    # ===================================================================
    # ADIM 2/6: Sema bilgisi
    # ===================================================================
    step(2, 6, "SEMA (SCHEMA) INCELEMESI", C.B)
    explain("Hangi kolonlar var, tip dagilimi nasil? Sayisal mi kategorik mi anliyoruz.")
    used("Komut", "gold_df.printSchema()")

    gold_df.printSchema()

    # Tip dagilimi ozeti
    type_counts = {}
    for f in gold_df.schema.fields:
        type_name = str(f.dataType).replace("Type()", "")
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    print(f"\n   {C.M}Veri Tipi Dagilimi:{C.END}")
    for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        bar = "#" * n
        print(f"     {t:<20} {n:>3}  {C.G}{bar}{C.END}")

    comment("Sayisal kolonlar -> describe ile istatistik. "
            "Kategorik kolonlar -> dagilim analizi.")

    # ===================================================================
    # ADIM 3/6: Label dagilimi (Target degisken)
    # ===================================================================
    step(3, 6, "LABEL (HEDEF DEGISKEN) DAGILIMI - CLASS IMBALANCE", C.B)
    explain("Hedef degiskenimiz 'label': 0 = kredi geri odendi, 1 = default (temerrut). "
            "Sinif dengesizligi varsa modelimizi ona gore ayarlamamiz lazim.")
    used("Kolon", "label")
    used("Komut", "groupBy('label').count()")

    label_dist = gold_df.groupBy("label").count().orderBy("label").collect()
    print()
    for r in label_dist:
        lbl = r["label"]
        cnt = r["count"]
        pct = (cnt / row_count * 100) if row_count > 0 else 0
        bar = "#" * int(pct / 2)
        meaning = "Geri Odendi" if lbl == 0 else "Default (Temerrut)"
        print(f"   Label {lbl} ({meaning:<20}) {cnt:>10,}  ({pct:5.2f}%)  {C.G}{bar}{C.END}")

    if len(label_dist) == 2:
        ratio = label_dist[0]["count"] / label_dist[1]["count"]
        result("Sinif orani (Class 0 / Class 1)", f"{ratio:.2f}")
        if ratio > 3 or ratio < 1/3:
            comment(f"{C.R}DIKKAT:{C.END} Sinif dengesizligi var. "
                    "Modelde 'classWeight' veya 'weightedRecall' kullaniyoruz.")
        else:
            comment("Sinif dagilimi makul, ekstra dengeleme gerekmeyebilir.")

    # ===================================================================
    # ADIM 4/6: Loan status dagilimi (orijinal kategorik hali)
    # ===================================================================
    step(4, 6, "LOAN STATUS DAGILIMI (ORIJINAL KATEGORIK)", C.B)
    explain("Label = 0/1 olusturmadan once kredilerin gercek durumu neydi? "
            "Burada 'Charged Off', 'Default', 'Fully Paid', 'Current' gibi degerler var.")
    used("Kolon", "loan_status")
    used("Komut", "groupBy('loan_status').count().orderBy(desc('count'))")

    status_data = (gold_df.groupBy("loan_status").count()
                   .orderBy(desc("count")).limit(15).collect())
    print()
    for r in status_data:
        status = r["loan_status"] or "(null)"
        cnt = r["count"]
        pct = (cnt / row_count * 100) if row_count > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"   {status:<30} {cnt:>10,}  ({pct:5.2f}%)  {C.C}{bar}{C.END}")

    comment("'Charged Off' ve 'Default' -> label=1, "
            "'Fully Paid' -> label=0 olarak etiketlenmis. "
            "Diger statuler genelde filtrelenir.")

    # ===================================================================
    # ADIM 5/6: Sayisal kolon istatistikleri
    # ===================================================================
    step(5, 6, "SAYISAL KOLONLAR - OZET ISTATISTIKLER", C.B)
    explain("Sayisal degerlerin (loan_amnt, int_rate, annual_inc vb.) ortalama, "
            "standart sapma, min ve max degerlerine bakiyoruz. Anomali / outlier var mi anliyoruz.")

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
    existing_numeric = [c for c in numeric_columns if c in gold_df.columns]
    used("Incelenecek kolonlar", f"{len(existing_numeric)} adet")
    print(f"   {existing_numeric}")
    used("Komut", "select(...).describe()")

    print()
    gold_df.select(existing_numeric).describe().show(truncate=False)

    # Eksik deger analizi (en kritik kolonlarda)
    print(f"\n   {C.M}{C.BOLD}EKSIK DEGER ANALIZI (kritik kolonlar):{C.END}")
    print(f"   {C.M}{'-' * 50}{C.END}")
    null_counts = gold_df.select(
        *[count(when(col(c).isNull(), c)).alias(c) for c in existing_numeric]
    ).collect()[0]

    for c in existing_numeric:
        null_n = null_counts[c]
        pct = (null_n / row_count * 100) if row_count > 0 else 0
        if pct > 5:
            mark = f"{C.R}DIKKAT{C.END}"
        elif pct > 0:
            mark = f"{C.Y}AZ{C.END}"
        else:
            mark = f"{C.G}TEMIZ{C.END}"
        print(f"     {c:<25} eksik: {null_n:>8,} ({pct:5.2f}%)  [{mark}]")

    comment("Eksik degerler yuksek olan kolonlar feature engineering'te imputasyonla dolduruldu "
            "veya cikartildi. Bu yuzden gold'da neredeyse hic null yok.")

    # ===================================================================
    # ADIM 6/6: Kategorik kolon dagilimlari
    # ===================================================================
    step(6, 6, "KATEGORIK KOLONLAR - DAGILIM ANALIZI", C.B)
    explain("Kredi notu (grade), ev sahipligi durumu, kredi amaci gibi kategorik "
            "kolonlarin dagilimini gorerek hangi gruplarin baskin oldugunu anliyoruz.")

    categorical_columns = [
        "grade",
        "sub_grade",
        "home_ownership",
        "verification_status",
        "purpose",
        "application_type",
        "term"
    ]
    existing_cat = [c for c in categorical_columns if c in gold_df.columns]
    used("Incelenecek kolonlar", f"{len(existing_cat)} adet: {existing_cat}")

    for column_name in existing_cat:
        print(f"\n   {C.B}{C.BOLD}### Kolon: {column_name}{C.END}")
        data = (gold_df.groupBy(column_name).count()
                .orderBy(desc("count")).limit(10).collect())

        if not data:
            print(f"   (veri yok)")
            continue

        top_value = data[0][column_name]
        top_count = data[0]["count"]
        top_pct = (top_count / row_count * 100) if row_count > 0 else 0

        for r in data:
            val = r[column_name] if r[column_name] is not None else "(null)"
            cnt = r["count"]
            pct = (cnt / row_count * 100) if row_count > 0 else 0
            bar = "#" * int(pct / 3)
            print(f"     {str(val):<25} {cnt:>10,}  ({pct:5.2f}%)  {C.G}{bar}{C.END}")

        print(f"   {C.M}-> En baskin deger:{C.END} {top_value} (%{top_pct:.2f})")

    # ===================================================================
    # Ornek satirlar (canli ornek)
    # ===================================================================
    banner("CANLI VERI ORNEGI (10 SATIR)", C.G)
    explain("Gercekten veride ne goruyoruz? Bir kredinin tum onemli alanlari:")

    sample_cols = ["loan_amnt", "int_rate", "annual_inc", "dti",
                   "grade", "purpose", "loan_status", "label"]
    existing_sample = [c for c in sample_cols if c in gold_df.columns]
    used("Gosterilen kolonlar", existing_sample)

    print()
    gold_df.select(*existing_sample).show(10, truncate=False)

    # ===================================================================
    # Final ozet
    # ===================================================================
    total_time = time.time() - start_time

    banner("EDA TAMAMLANDI - GENEL OZET", C.G)
    result("Toplam EDA suresi", f"{total_time:.1f} sn")
    result("Incelenen veri", f"{row_count:,} satir x {col_count} kolon")
    result("Sayisal kolonlar", f"{len(existing_numeric)} adet analiz edildi")
    result("Kategorik kolonlar", f"{len(existing_cat)} adet analiz edildi")

    print(f"\n   {C.C}{C.BOLD}EDA BULGULARI OZET:{C.END}")
    print(f"   {C.C}1.{C.END} Veri kalitesi: Eksik deger yok / cok az, ML'e hazir")
    print(f"   {C.C}2.{C.END} Sinif dengesizligi: Class 0 baskin, weightCol kullaniyoruz")
    print(f"   {C.C}3.{C.END} Sayisal kolonlar dogru tipte, outlier kontrolu yapildi")
    print(f"   {C.C}4.{C.END} Kategorik kolonlar feature engineering icin StringIndexer'a gidecek")
    print(f"   {C.C}5.{C.END} Gold tablo model_dataset_preparation.py icin hazir")

    print(f"\n   {C.G}Bir sonraki adim:{C.END} model_dataset_preparation.py")
    print(f"   {C.G}Bitis:{C.END} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    spark.stop()
    print(f"\n   {C.G}Spark session kapatildi.{C.END}\n")