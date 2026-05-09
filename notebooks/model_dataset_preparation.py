from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ---------------------------------------------------------------
# 1. Spark Session
# ---------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("LendingClub_ModelDatasetPrep")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
print("✅ Spark session başlatıldı:", spark.version)

# ---------------------------------------------------------------
# 2. Gold Katmanını Oku
# ---------------------------------------------------------------
GOLD_PATH = "/home/jovyan/work/lending_club_delta/gold"

gold_df = spark.read.format("delta").load(GOLD_PATH)

print(f"📊 Gold satır sayısı: {gold_df.count():,}")
print(f"📊 Gold kolon sayısı: {len(gold_df.columns)}")
print("\n📋 Label dağılımı:")
gold_df.groupBy("label").count().orderBy("label").show()

# ---------------------------------------------------------------
# 3. Data Leakage Kolonlarını Drop Et
# ---------------------------------------------------------------
# Bu kolonlar kredi VERİLDİKTEN SONRA dolar.
# Yeni başvuru anında bu bilgiler henüz mevcut değildir,
# bu yüzden modele dahil edilmesi data leakage'a yol açar.
leakage_cols = [
    # Ödeme ve tahsilat bilgileri (kredi verildikten sonra)
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",

    # Çıkış bilgileri (kredi kapandıktan sonra)
    "out_prncp", "out_prncp_inv",

    # loan_status zaten label'a dönüştü, kalmamalı
    "loan_status",

    # Tahsilat ile ilgili dolaylı kolonlar
    "collections_12_mths_ex_med",
]

# Sadece DataFrame'de gerçekten var olan kolonları drop et
existing_leakage = [c for c in leakage_cols if c in gold_df.columns]
print(f"\n🚫 Drop edilecek leakage kolonları ({len(existing_leakage)} adet):")
for c in existing_leakage:
    print(f"   - {c}")

df_no_leakage = gold_df.drop(*existing_leakage)
print(f"\n📊 Leakage drop sonrası kolon sayısı: {len(df_no_leakage.columns)}")

# ---------------------------------------------------------------
# 4. Feature Seçimi (20-30 kolon)
# ---------------------------------------------------------------
# Sayısal Features (başvuru anında bilinen bilgiler)
numeric_features = [
    "loan_amnt",          # Talep edilen kredi tutarı
    "int_rate",           # Faiz oranı
    "installment",        # Aylık taksit
    "annual_inc",         # Yıllık gelir
    "dti",                # Borç/Gelir oranı (debt-to-income)
    "delinq_2yrs",        # Son 2 yıldaki gecikme sayısı
    "fico_range_low",     # FICO kredi notu (alt)
    "fico_range_high",    # FICO kredi notu (üst)
    "open_acc",           # Açık kredi hesap sayısı
    "pub_rec",            # Kamu kayıtlarındaki olumsuz kayıt
    "revol_bal",          # Toplam kredi kartı bakiyesi
    "revol_util",         # Kredi kartı kullanım oranı (%)
    "total_acc",          # Toplam kredi hesap sayısı
    "mort_acc",           # Mortgage hesap sayısı
    "pub_rec_bankruptcies", # İflas kaydı sayısı
]

# Kategorik Features (başvuru anında bilinen bilgiler)
categorical_features = [
    "term",               # Kredi vadesi (36 ay / 60 ay)
    "grade",              # Kredi notu (A-G)
    "sub_grade",           # Alt kredi notu (A1-G5)
    "emp_length",         # İş tecrübesi süresi
    "home_ownership",     # Ev sahipliği durumu
    "verification_status", # Gelir doğrulama durumu
    "purpose",            # Kredi amacı
    "application_type",   # Başvuru tipi (Bireysel / Ortak)
    "initial_list_status", # İlk liste durumu
]

target_col = "label"

# Sadece DataFrame'de var olan kolonları seç
numeric_features = [c for c in numeric_features if c in df_no_leakage.columns]
categorical_features = [c for c in categorical_features if c in df_no_leakage.columns]

print(f"\n🔢 Sayısal feature sayısı: {len(numeric_features)}")
print(f"🔤 Kategorik feature sayısı: {len(categorical_features)}")
print(f"🎯 Toplam feature sayısı: {len(numeric_features) + len(categorical_features)}")

# Sadece seçilen kolonları içeren DataFrame
selected_cols = numeric_features + categorical_features + [target_col]
df_selected = df_no_leakage.select(*selected_cols)

print(f"\n✅ Seçilen DataFrame:")
print(f"   - Satır sayısı: {df_selected.count():,}")
print(f"   - Kolon sayısı: {len(df_selected.columns)}")

print("\n📋 İlk 5 satır:")
df_selected.show(5, truncate=False)

print("\n📋 Şema:")
df_selected.printSchema()

# ---------------------------------------------------------------
# 5. Kategorik Kolonları Encode Et
# ---------------------------------------------------------------
from pyspark.ml.feature import StringIndexer, OneHotEncoder

print("\n" + "="*60)
print("KATEGORİK KOLON ENCODING")
print("="*60)

# Her kategorik kolon için:
#   <kolon> -> <kolon>_idx (StringIndexer çıktısı)
#   <kolon>_idx -> <kolon>_ohe (OneHotEncoder çıktısı)

indexers = [
    StringIndexer(
        inputCol=col,
        outputCol=f"{col}_idx",
        handleInvalid="keep"  # Bilinmeyen değer gelirse hata vermesin
    )
    for col in categorical_features
]

encoders = [
    OneHotEncoder(
        inputCol=f"{col}_idx",
        outputCol=f"{col}_ohe"
    )
    for col in categorical_features
]

print(f"\n✅ {len(indexers)} adet StringIndexer oluşturuldu")
print(f"✅ {len(encoders)} adet OneHotEncoder oluşturuldu")

# Henüz fit etmedik — sadece transformer'ları hazırladık.
# Pipeline'a koyup hep birlikte fit edeceğiz (BLOK 5'te).

# Encoded kolon isimlerini sakla (sonraki blokta lazım olacak)
ohe_cols = [f"{col}_ohe" for col in categorical_features]

print(f"\n📋 OneHotEncoded kolon isimleri:")
for c in ohe_cols:
    print(f"   - {c}")

# ---------------------------------------------------------------
# 6. Sayısal Kolonları Birleştir + Scale Et
# ---------------------------------------------------------------
from pyspark.ml.feature import VectorAssembler, StandardScaler

print("\n" + "="*60)
print("VECTORASSEMBLER + STANDARDSCALER")
print("="*60)

# 6a. Önce sadece sayısal kolonları tek vektörde birleştir
numeric_assembler = VectorAssembler(
    inputCols=numeric_features,
    outputCol="numeric_vec",
    handleInvalid="keep"
)

# 6b. Sayısal vektörü scale et (mean=0, std=1)
scaler = StandardScaler(
    inputCol="numeric_vec",
    outputCol="numeric_scaled",
    withMean=True,
    withStd=True
)

print(f"✅ VectorAssembler hazırlandı: {len(numeric_features)} sayısal kolon → numeric_vec")
print(f"✅ StandardScaler hazırlandı: numeric_vec → numeric_scaled")

# 6c. Final assembler — scaled sayısal + OHE kategorik kolonları tek vektörde birleştir
final_assembler = VectorAssembler(
    inputCols=["numeric_scaled"] + ohe_cols,
    outputCol="features",
    handleInvalid="keep"
)

print(f"✅ Final VectorAssembler hazırlandı: numeric_scaled + {len(ohe_cols)} OHE → features")

# ---------------------------------------------------------------
# 7. Class Weight Hesapla (Imbalance için)
# ---------------------------------------------------------------
print("\n" + "="*60)
print("CLASS WEIGHT HESAPLAMA")
print("="*60)

# Label sayılarını al
label_counts = df_selected.groupBy("label").count().collect()
total = sum(row["count"] for row in label_counts)
n_classes = len(label_counts)

# weight = total / (n_classes * count_of_class)
# Bu standart formül: az sayıdaki sınıfa yüksek ağırlık verir
class_weights = {}
for row in label_counts:
    class_weights[row["label"]] = total / (n_classes * row["count"])

print(f"📊 Toplam satır: {total:,}")
print(f"📊 Sınıf sayısı: {n_classes}")
print(f"\n⚖️  Hesaplanan class weights:")
for label_val, weight in sorted(class_weights.items()):
    print(f"   - label={label_val}: weight={weight:.4f}")

# classWeight kolonunu DataFrame'e ekle
weight_expr = F.when(F.col("label") == 0, class_weights[0]) \
                .otherwise(class_weights[1])

df_with_weight = df_selected.withColumn("classWeight", weight_expr)

print(f"\n✅ classWeight kolonu eklendi")
print(f"📊 Yeni kolon sayısı: {len(df_with_weight.columns)}")

# Kontrol: ağırlıkların doğru atandığını gör
print("\n📋 Class weight örneği (her sınıftan 3 satır):")
df_with_weight.select("label", "classWeight").distinct().orderBy("label").show()

# ---------------------------------------------------------------
# 8. Pipeline'ı Kur ve Fit Et
# ---------------------------------------------------------------
from pyspark.ml import Pipeline

print("\n" + "="*60)
print("PIPELINE FIT & TRANSFORM")
print("="*60)

# Tüm aşamaları sırayla pipeline'a ver:
# 1. Önce StringIndexer'lar (string -> int)
# 2. Sonra OneHotEncoder'lar (int -> one-hot vector)
# 3. Sonra numeric_assembler (sayısalları topla)
# 4. Sonra scaler (sayısalları normalize et)
# 5. Son olarak final_assembler (her şeyi tek vektörde birleştir)
pipeline_stages = indexers + encoders + [numeric_assembler, scaler, final_assembler]

print(f"📦 Pipeline {len(pipeline_stages)} aşamadan oluşuyor:")
for i, stage in enumerate(pipeline_stages, 1):
    print(f"   {i}. {type(stage).__name__}")

pipeline = Pipeline(stages=pipeline_stages)

# Fit (Pipeline veriden öğrenir: hangi kategoriler var, ortalama/std ne)
print("\n⏳ Pipeline fit ediliyor (1-2 dakika sürebilir)...")
pipeline_model = pipeline.fit(df_with_weight)
print("✅ Pipeline fit edildi")

# Transform (Veriyi pipeline'dan geçir)
print("\n⏳ Veri transform ediliyor...")
df_transformed = pipeline_model.transform(df_with_weight)

# Sadece final ihtiyacımız olan kolonları tut: features + label + classWeight
df_final = df_transformed.select("features", "label", "classWeight")

print("✅ Veri transform edildi")
print(f"\n📊 Final dataset:")
print(f"   - Satır sayısı: {df_final.count():,}")
print(f"   - Kolonlar: {df_final.columns}")

print("\n📋 İlk 3 satır:")
df_final.show(3, truncate=80)

# ---------------------------------------------------------------
# 9. Stratified Train/Test Split (80/20)
# ---------------------------------------------------------------
print("\n" + "="*60)
print("TRAIN/TEST SPLIT (80/20, STRATIFIED)")
print("="*60)

# sampleBy: her sınıftan ayrı ayrı %80 örneklem alır → stratified
fractions = {0: 0.8, 1: 0.8}
train_df = df_final.sampleBy("label", fractions=fractions, seed=42)

# Test = full - train (Spark'ta direkt subtract kullanıyoruz)
# Ama subtract pahalı olduğu için, başka bir yaklaşım: monotonic id ile işaretleme
# Daha basit: train'i cache'le, test'i except() ile al
train_df = train_df.cache()
test_df = df_final.subtract(train_df)

train_count = train_df.count()
test_count = test_df.count()

print(f"📊 Train satır sayısı: {train_count:,}")
print(f"📊 Test satır sayısı: {test_count:,}")
print(f"📊 Toplam: {train_count + test_count:,}")

print("\n📋 Train label dağılımı:")
train_df.groupBy("label").count().orderBy("label").show()

print("📋 Test label dağılımı:")
test_df.groupBy("label").count().orderBy("label").show()

# ---------------------------------------------------------------
# 10. Delta Lake'e Yaz (ml_ready katmanı)
# ---------------------------------------------------------------
print("\n" + "="*60)
print("DELTA LAKE'E YAZMA (ml_ready)")
print("="*60)

ML_READY_PATH = "/home/jovyan/work/lending_club_delta/ml_ready"
TRAIN_PATH = f"{ML_READY_PATH}/train"
TEST_PATH = f"{ML_READY_PATH}/test"
PIPELINE_MODEL_PATH = f"{ML_READY_PATH}/pipeline_model"

print(f"⏳ Train seti yazılıyor: {TRAIN_PATH}")
train_df.write.format("delta").mode("overwrite").save(TRAIN_PATH)
print("✅ Train seti yazıldı")

print(f"\n⏳ Test seti yazılıyor: {TEST_PATH}")
test_df.write.format("delta").mode("overwrite").save(TEST_PATH)
print("✅ Test seti yazıldı")

print(f"\n⏳ Pipeline modeli kaydediliyor: {PIPELINE_MODEL_PATH}")
pipeline_model.write().overwrite().save(PIPELINE_MODEL_PATH)
print("✅ Pipeline modeli kaydedildi")

# ---------------------------------------------------------------
# 11. Final Özet
# ---------------------------------------------------------------
print("\n" + "="*60)
print("🎉 MODEL DATASET PREPARATION TAMAMLANDI")
print("="*60)
print(f"""
📂 Çıktılar:
   - Train Delta:    {TRAIN_PATH}
   - Test Delta:     {TEST_PATH}
   - Pipeline Model: {PIPELINE_MODEL_PATH}

📊 İstatistikler:
   - Train: {train_count:,} satır
   - Test:  {test_count:,} satır
   - Feature sayısı: {len(numeric_features) + len(categorical_features)} (encode öncesi)
   - Class weight: label=0 → 0.5676, label=1 → 4.1964

🚀 Sıradaki Adım:
   notebooks/model_training.py
   → Logistic Regression, Random Forest, GBT eğitimi + MLflow
""")

spark.stop()