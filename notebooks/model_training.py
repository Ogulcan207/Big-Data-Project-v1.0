
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ---------------------------------------------------------------
# 1. Spark Session
# ---------------------------------------------------------------
spark = (
    SparkSession.builder
    .appName("LendingClub_ModelTraining")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "2g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
print("✅ Spark session başlatıldı:", spark.version)

# ---------------------------------------------------------------
# 2. MLflow Setup
# ---------------------------------------------------------------
import mlflow
import mlflow.spark

# Tracking URI: tüm deneyler bu klasöre kaydedilecek
MLFLOW_TRACKING_URI = "file:///home/jovyan/work/mlruns"
EXPERIMENT_NAME = "LendingClub_DefaultPrediction"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"✅ MLflow tracking URI: {MLFLOW_TRACKING_URI}")
print(f"✅ MLflow experiment: {EXPERIMENT_NAME}")

# ---------------------------------------------------------------
# 3. Train/Test Setlerini Yükle
# ---------------------------------------------------------------
ML_READY_PATH = "/home/jovyan/work/lending_club_delta/ml_ready"
TRAIN_PATH = f"{ML_READY_PATH}/train"
TEST_PATH = f"{ML_READY_PATH}/test"

print("\n⏳ Train/Test setleri yükleniyor...")
train_df = spark.read.format("delta").load(TRAIN_PATH).cache()
test_df = spark.read.format("delta").load(TEST_PATH).cache()

train_count = train_df.count()  # cache'i tetikle
test_count = test_df.count()

print(f"✅ Train: {train_count:,} satır")
print(f"✅ Test:  {test_count:,} satır")

print("\n📋 Train şeması:")
train_df.printSchema()

print("📋 Train label dağılımı:")
train_df.groupBy("label").count().orderBy("label").show()

print("📋 Test label dağılımı:")
test_df.groupBy("label").count().orderBy("label").show()

print("\n🎉 BLOK 1 tamamlandı — modeller eğitilmeye hazır!")

# ---------------------------------------------------------------
# 4. Yardımcı Fonksiyonlar
# ---------------------------------------------------------------
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)


def compute_confusion_matrix(predictions_df):
    """
    Spark DataFrame üzerinden confusion matrix hesaplar.
    Returns: (TN, FP, FN, TP)
    """
    # label vs prediction çapraz tablo
    cm = predictions_df.groupBy("label", "prediction").count().collect()

    tn = fp = fn = tp = 0
    for row in cm:
        label, pred, count = row["label"], row["prediction"], row["count"]
        if label == 0 and pred == 0.0:
            tn = count
        elif label == 0 and pred == 1.0:
            fp = count
        elif label == 1 and pred == 0.0:
            fn = count
        elif label == 1 and pred == 1.0:
            tp = count

    return tn, fp, fn, tp


def compute_all_metrics(predictions_df):
    """
    Tüm sınıflandırma metriklerini hesaplar.
    Returns: dict {auc, accuracy, f1, precision, recall, tn, fp, fn, tp}
    """
    # Binary metrikler (AUC için)
    binary_eval = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    auc = binary_eval.evaluate(predictions_df)

    # Multiclass metrikler (Accuracy, F1, Precision, Recall)
    accuracy_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    precision_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="weightedPrecision"
    )
    recall_eval = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="weightedRecall"
    )

    accuracy = accuracy_eval.evaluate(predictions_df)
    f1 = f1_eval.evaluate(predictions_df)
    precision = precision_eval.evaluate(predictions_df)
    recall = recall_eval.evaluate(predictions_df)

    # Confusion Matrix
    tn, fp, fn, tp = compute_confusion_matrix(predictions_df)

    return {
        "auc": auc,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def train_and_evaluate(model, model_name, params, train_df, test_df,
                        log_feature_importance=False):
    """
    Modeli eğitir, değerlendirir ve MLflow'a loglar.

    Args:
        model: Spark ML model instance (henüz fit edilmemiş)
        model_name: MLflow run adı (str)
        params: Loglanacak hiperparametreler (dict)
        train_df: Eğitim seti
        test_df: Test seti
        log_feature_importance: Tree-based modeller için True

    Returns:
        dict: model + metrics
    """
    print(f"\n{'='*60}")
    print(f"🚀 {model_name} eğitiliyor...")
    print(f"{'='*60}")

    with mlflow.start_run(run_name=model_name):
        # Parametreleri logla
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("train_size", train_df.count())
        mlflow.log_param("test_size", test_df.count())

        # ---- Train ----
        start_time = time.time()
        fitted_model = model.fit(train_df)
        train_time = time.time() - start_time

        print(f"⏱️  Eğitim süresi: {train_time:.2f} saniye")
        mlflow.log_metric("train_time_seconds", train_time)

        # ---- Predict ----
        predictions = fitted_model.transform(test_df)

        # ---- Metrics ----
        metrics = compute_all_metrics(predictions)

        # MLflow'a metrikleri logla
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

        # ---- Print ----
        print(f"\n📊 Test Metrikleri:")
        print(f"   AUC-ROC:   {metrics['auc']:.4f}")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   F1-Score:  {metrics['f1']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")

        print(f"\n📋 Confusion Matrix:")
        print(f"                  Predicted 0   Predicted 1")
        print(f"   Actual 0:       {metrics['tn']:>10,}    {metrics['fp']:>10,}")
        print(f"   Actual 1:       {metrics['fn']:>10,}    {metrics['tp']:>10,}")

        # ---- Feature Importance (tree-based modellerde) ----
        if log_feature_importance and hasattr(fitted_model, "featureImportances"):
            importances = fitted_model.featureImportances.toArray().tolist()
            print(f"\n🌟 Feature Importance (top 10):")
            top_indices = sorted(
                range(len(importances)),
                key=lambda i: importances[i],
                reverse=True
            )[:10]
            for rank, idx in enumerate(top_indices, 1):
                print(f"   {rank}. Feature[{idx}]: {importances[idx]:.4f}")

            # Tüm önem skorlarını MLflow'a logla
            for i, imp in enumerate(importances):
                mlflow.log_metric(f"fi_feature_{i}", imp)

        # ---- Modeli MLflow'a kaydet ----
        try:
            mlflow.spark.log_model(fitted_model, "model")
            print(f"\n✅ Model MLflow'a kaydedildi")
        except Exception as e:
            print(f"\n⚠️  Model MLflow'a kaydedilemedi: {e}")

        return {
            "model_name": model_name,
            "fitted_model": fitted_model,
            "metrics": metrics,
            "train_time": train_time,
        }


# Sonuçları toplamak için liste
all_results = []

print("\n✅ Yardımcı fonksiyonlar yüklendi")
print("📦 Sıradaki blok: 5 model eğitimi")


# ---------------------------------------------------------------
# 5. Model 1: Logistic Regression
# ---------------------------------------------------------------
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",   # imbalance handling
    maxIter=50,
    regParam=0.01,
    elasticNetParam=0.0,        # L2 regularization (Ridge)
    family="binomial",
)

lr_params = {
    "maxIter": 50,
    "regParam": 0.01,
    "elasticNetParam": 0.0,
    "family": "binomial",
    "weightCol": "classWeight",
}

lr_result = train_and_evaluate(
    model=lr,
    model_name="LogisticRegression",
    params=lr_params,
    train_df=train_df,
    test_df=test_df,
    log_feature_importance=False,  # LR'da featureImportances yok
)
all_results.append(lr_result)


# ---------------------------------------------------------------
# 6. Model 2: Linear SVC
# ---------------------------------------------------------------
from pyspark.ml.classification import LinearSVC

svc = LinearSVC(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",
    maxIter=20,        # SVC yavaş, az iter yeterli
    regParam=0.01,
)

svc_params = {
    "maxIter": 20,
    "regParam": 0.01,
    "weightCol": "classWeight",
}

svc_result = train_and_evaluate(
    model=svc,
    model_name="LinearSVC",
    params=svc_params,
    train_df=train_df,
    test_df=test_df,
    log_feature_importance=False,
)
all_results.append(svc_result)


# ---------------------------------------------------------------
# 7. Model 3: Decision Tree Classifier
# ---------------------------------------------------------------
from pyspark.ml.classification import DecisionTreeClassifier

dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",
    maxDepth=10,        # Derin ağaç → overfitting riski, 10 makul
    maxBins=32,
    minInstancesPerNode=100,
)

dt_params = {
    "maxDepth": 10,
    "maxBins": 32,
    "minInstancesPerNode": 100,
    "weightCol": "classWeight",
}

dt_result = train_and_evaluate(
    model=dt,
    model_name="DecisionTree",
    params=dt_params,
    train_df=train_df,
    test_df=test_df,
    log_feature_importance=True,  # Tree-based → feature importance var
)
all_results.append(dt_result)


# ---------------------------------------------------------------
# 8. Model 4: Random Forest Classifier
# ---------------------------------------------------------------
from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",
    numTrees=50,           # 50 ağaç (100 daha iyi ama yavaş)
    maxDepth=10,
    maxBins=32,
    minInstancesPerNode=100,
    subsamplingRate=0.8,
    seed=42,
)

rf_params = {
    "numTrees": 50,
    "maxDepth": 10,
    "maxBins": 32,
    "minInstancesPerNode": 100,
    "subsamplingRate": 0.8,
    "weightCol": "classWeight",
}

rf_result = train_and_evaluate(
    model=rf,
    model_name="RandomForest",
    params=rf_params,
    train_df=train_df,
    test_df=test_df,
    log_feature_importance=True,
)
all_results.append(rf_result)


# ---------------------------------------------------------------
# 9. Model 5: Gradient Boosted Trees (GBT)
# ---------------------------------------------------------------
from pyspark.ml.classification import GBTClassifier

gbt = GBTClassifier(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",
    maxIter=30,            # 30 ağaç (50 daha iyi ama uzun sürer)
    maxDepth=6,
    maxBins=32,
    stepSize=0.1,           # Learning rate
    subsamplingRate=0.8,
    seed=42,
)

gbt_params = {
    "maxIter": 30,
    "maxDepth": 6,
    "maxBins": 32,
    "stepSize": 0.1,
    "subsamplingRate": 0.8,
    "weightCol": "classWeight",
}

gbt_result = train_and_evaluate(
    model=gbt,
    model_name="GBTClassifier",
    params=gbt_params,
    train_df=train_df,
    test_df=test_df,
    log_feature_importance=True,
)
all_results.append(gbt_result)


# ---------------------------------------------------------------
# 10. Özet
# ---------------------------------------------------------------
print("\n" + "="*70)
print("🎉 TÜM MODELLER EĞİTİLDİ — ÖZET")
print("="*70)
print(f"\n{'Model':<20} {'AUC':>8} {'Accuracy':>10} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Süre(s)':>10}")
print("-" * 76)
for result in all_results:
    m = result["metrics"]
    print(f"{result['model_name']:<20} "
          f"{m['auc']:>8.4f} "
          f"{m['accuracy']:>10.4f} "
          f"{m['f1']:>8.4f} "
          f"{m['precision']:>10.4f} "
          f"{m['recall']:>8.4f} "
          f"{result['train_time']:>10.1f}")

# En iyi model (AUC'ye göre)
best_result = max(all_results, key=lambda r: r["metrics"]["auc"])
print(f"\n🏆 En İyi Model (AUC bazlı): {best_result['model_name']}")
print(f"   AUC: {best_result['metrics']['auc']:.4f}")

# ---------------------------------------------------------------
# 11. Sonuçları DataFrame'e Çevir ve Kaydet
# ---------------------------------------------------------------
print("\n" + "="*70)
print("📊 SONUÇLARI KAYDETME")
print("="*70)

# all_results listesinden DataFrame oluştur
results_data = []
for r in all_results:
    m = r["metrics"]
    results_data.append({
        "model_name": r["model_name"],
        "auc": float(m["auc"]),
        "accuracy": float(m["accuracy"]),
        "f1": float(m["f1"]),
        "precision": float(m["precision"]),
        "recall": float(m["recall"]),
        "tn": int(m["tn"]),
        "fp": int(m["fp"]),
        "fn": int(m["fn"]),
        "tp": int(m["tp"]),
        "train_time_seconds": float(r["train_time"]),
    })

# Spark DataFrame oluştur
results_df = spark.createDataFrame(results_data)

# AUC'ye göre sırala (büyükten küçüğe)
results_df = results_df.orderBy(F.col("auc").desc())

print("\n📋 Sonuç DataFrame'i:")
results_df.show(truncate=False)

# ---------------------------------------------------------------
# 12. Sonuçları Delta'ya Kaydet
# ---------------------------------------------------------------
RESULTS_PATH = "/home/jovyan/work/lending_club_delta/model_results"

print(f"\n⏳ Sonuçlar Delta'ya yazılıyor: {RESULTS_PATH}")
results_df.write.format("delta").mode("overwrite").save(RESULTS_PATH)
print("✅ Sonuçlar Delta'ya kaydedildi")

# CSV olarak da kaydet (dashboard'da kolay okumak için)
RESULTS_CSV_PATH = "/home/jovyan/work/lending_club_delta/model_results_csv"
print(f"\n⏳ Sonuçlar CSV'ye yazılıyor: {RESULTS_CSV_PATH}")
(
    results_df
    .coalesce(1)  # tek dosyada olsun
    .write.mode("overwrite")
    .option("header", True)
    .csv(RESULTS_CSV_PATH)
)
print("✅ Sonuçlar CSV'ye kaydedildi")

# ---------------------------------------------------------------
# 13. En İyi Modeli Production Path'e Kaydet
# ---------------------------------------------------------------
best_result = max(all_results, key=lambda r: r["metrics"]["auc"])
best_model = best_result["fitted_model"]
best_model_name = best_result["model_name"]

BEST_MODEL_PATH = "/home/jovyan/work/lending_club_delta/best_model"

print(f"\n⏳ En iyi model ({best_model_name}) kaydediliyor: {BEST_MODEL_PATH}")
best_model.write().overwrite().save(BEST_MODEL_PATH)
print(f"✅ En iyi model kaydedildi")

# ---------------------------------------------------------------
# 14. Final Özet
# ---------------------------------------------------------------
print("\n" + "="*70)
print("🎉 MODEL TRAINING TAMAMEN TAMAMLANDI")
print("="*70)
print(f"""
📂 Çıktılar:
   - Sonuç tablosu (Delta): {RESULTS_PATH}
   - Sonuç tablosu (CSV):   {RESULTS_CSV_PATH}
   - En iyi model:          {BEST_MODEL_PATH}
   - MLflow runs:           {MLFLOW_TRACKING_URI}

🏆 En İyi Model: {best_model_name}
   - AUC:       {best_result['metrics']['auc']:.4f}
   - Accuracy:  {best_result['metrics']['accuracy']:.4f}
   - F1:        {best_result['metrics']['f1']:.4f}
   - Precision: {best_result['metrics']['precision']:.4f}
   - Recall:    {best_result['metrics']['recall']:.4f}

📊 Tüm Modeller (AUC sırasına göre):
""")
for i, r in enumerate(sorted(all_results, key=lambda x: x["metrics"]["auc"], reverse=True), 1):
    m = r["metrics"]
    print(f"   {i}. {r['model_name']:<22} AUC={m['auc']:.4f}  F1={m['f1']:.4f}  Süre={r['train_time']:.1f}s")

print(f"""
🚀 Sıradaki Adım:
   - notebooks/dashboard.py veya .ipynb (görselleştirme + dashboard)
   - MLflow UI: deneyleri görsel olarak incelemek için
""")

spark.stop()
print("✅ Spark session kapatıldı")