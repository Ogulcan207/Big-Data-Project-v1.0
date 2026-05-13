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
# 2. ÇIKTI KLASÖRÜ — Proje kökünde result/
# ---------------------------------------------------------------
RESULT_PATH = "/home/jovyan/work/result"
os.makedirs(RESULT_PATH, exist_ok=True)
print(f"✅ Çıktı klasörü hazır: {RESULT_PATH}")

# ---------------------------------------------------------------
# 3. MLflow Setup (Şartname zorunlu kılıyor — %15 puan)
# ---------------------------------------------------------------
import mlflow
import mlflow.spark

MLFLOW_TRACKING_URI = "file:///home/jovyan/work/mlruns"
EXPERIMENT_NAME = "LendingClub_DefaultPrediction"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"✅ MLflow tracking URI: {MLFLOW_TRACKING_URI}")
print(f"✅ MLflow experiment: {EXPERIMENT_NAME}")

# ---------------------------------------------------------------
# 4. Train/Test Setlerini Yükle
# ---------------------------------------------------------------
ML_READY_PATH = "/home/jovyan/work/lending_club_delta/ml_ready"
TRAIN_PATH = f"{ML_READY_PATH}/train"
TEST_PATH = f"{ML_READY_PATH}/test"
GOLD_PATH = "/home/jovyan/work/lending_club_delta/gold"

print("\n⏳ Train/Test setleri yükleniyor...")
train_df = spark.read.format("delta").load(TRAIN_PATH).cache()
test_df = spark.read.format("delta").load(TEST_PATH).cache()

train_count = train_df.count()
test_count = test_df.count()

print(f"✅ Train: {train_count:,} satır")
print(f"✅ Test:  {test_count:,} satır")

print("\n📋 Train label dağılımı:")
train_df.groupBy("label").count().orderBy("label").show()

gold_df = None
try:
    gold_df = spark.read.format("delta").load(GOLD_PATH).cache()
    print(f"✅ Gold tablo yüklendi: {gold_df.count():,} satır")
except Exception as e:
    print(f"⚠️  Gold tablo yüklenemedi (train_df kullanılacak): {e}")

# ---------------------------------------------------------------
# 5. Yardımcı Fonksiyonlar
# ---------------------------------------------------------------
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)


def compute_confusion_matrix(predictions_df):
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
    auc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction",
        metricName="areaUnderROC").evaluate(predictions_df)
    accuracy = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="accuracy").evaluate(predictions_df)
    f1 = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="f1").evaluate(predictions_df)
    precision = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="weightedPrecision").evaluate(predictions_df)
    recall = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction",
        metricName="weightedRecall").evaluate(predictions_df)
    tn, fp, fn, tp = compute_confusion_matrix(predictions_df)
    return {"auc": auc, "accuracy": accuracy, "f1": f1,
            "precision": precision, "recall": recall,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def train_and_evaluate(model, model_name, params, train_df, test_df,
                       log_feature_importance=False):
    print(f"\n{'='*60}")
    print(f"🚀 {model_name} eğitiliyor...")
    print(f"{'='*60}")

    with mlflow.start_run(run_name=model_name):
        # MLflow: parametreleri logla
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("train_size", train_count)
        mlflow.log_param("test_size", test_count)

        start_time = time.time()
        fitted_model = model.fit(train_df)
        train_time = time.time() - start_time
        print(f"⏱️  Eğitim süresi: {train_time:.2f} saniye")
        mlflow.log_metric("train_time_seconds", train_time)

        predictions = fitted_model.transform(test_df)
        metrics = compute_all_metrics(predictions)

        # MLflow: metrikleri logla
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)

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

        feature_importances = None
        if log_feature_importance and hasattr(fitted_model, "featureImportances"):
            feature_importances = fitted_model.featureImportances.toArray().tolist()
            print(f"\n🌟 Feature Importance (top 10):")
            top_indices = sorted(range(len(feature_importances)),
                                  key=lambda i: feature_importances[i],
                                  reverse=True)[:10]
            for rank, idx in enumerate(top_indices, 1):
                print(f"   {rank}. Feature[{idx}]: {feature_importances[idx]:.4f}")
            for i, imp in enumerate(feature_importances):
                mlflow.log_metric(f"fi_feature_{i}", imp)

        # MLflow: modeli kaydet
        try:
            mlflow.spark.log_model(fitted_model, "model")
            print(f"\n✅ Model MLflow'a kaydedildi")
        except Exception as e:
            print(f"\n⚠️  Model MLflow'a kaydedilemedi: {e}")

        return {
            "model_name": model_name,
            "fitted_model": fitted_model,
            "predictions": predictions,
            "metrics": metrics,
            "train_time": train_time,
            "feature_importances": feature_importances,
        }


all_results = []
print("\n✅ Yardımcı fonksiyonlar yüklendi")

# ---------------------------------------------------------------
# 6-10. 5 Model Eğitimi
# ---------------------------------------------------------------
from pyspark.ml.classification import (
    LogisticRegression, LinearSVC, DecisionTreeClassifier,
    RandomForestClassifier, GBTClassifier,
)

# Model 1: Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label",
                         weightCol="classWeight", maxIter=50, regParam=0.01,
                         elasticNetParam=0.0, family="binomial")
lr_params = {"maxIter": 50, "regParam": 0.01, "elasticNetParam": 0.0,
             "family": "binomial", "weightCol": "classWeight"}
all_results.append(train_and_evaluate(lr, "LogisticRegression",
                                       lr_params, train_df, test_df, False))

# Model 2: Linear SVC
svc = LinearSVC(featuresCol="features", labelCol="label",
                 weightCol="classWeight", maxIter=20, regParam=0.01)
svc_params = {"maxIter": 20, "regParam": 0.01, "weightCol": "classWeight"}
all_results.append(train_and_evaluate(svc, "LinearSVC",
                                       svc_params, train_df, test_df, False))

# Model 3: Decision Tree
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label",
                             weightCol="classWeight", maxDepth=10,
                             maxBins=32, minInstancesPerNode=100)
dt_params = {"maxDepth": 10, "maxBins": 32, "minInstancesPerNode": 100,
             "weightCol": "classWeight"}
all_results.append(train_and_evaluate(dt, "DecisionTree",
                                       dt_params, train_df, test_df, True))

# Model 4: Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label",
                             weightCol="classWeight", numTrees=50, maxDepth=10,
                             maxBins=32, minInstancesPerNode=100,
                             subsamplingRate=0.8, seed=42)
rf_params = {"numTrees": 50, "maxDepth": 10, "maxBins": 32,
             "minInstancesPerNode": 100, "subsamplingRate": 0.8,
             "weightCol": "classWeight"}
all_results.append(train_and_evaluate(rf, "RandomForest",
                                       rf_params, train_df, test_df, True))

# Model 5: GBT
gbt = GBTClassifier(featuresCol="features", labelCol="label",
                     weightCol="classWeight", maxIter=30, maxDepth=6,
                     maxBins=32, stepSize=0.1, subsamplingRate=0.8, seed=42)
gbt_params = {"maxIter": 30, "maxDepth": 6, "maxBins": 32, "stepSize": 0.1,
              "subsamplingRate": 0.8, "weightCol": "classWeight"}
all_results.append(train_and_evaluate(gbt, "GBTClassifier",
                                       gbt_params, train_df, test_df, True))

# ---------------------------------------------------------------
# 11. Özet
# ---------------------------------------------------------------
print("\n" + "="*70)
print("🎉 TÜM MODELLER EĞİTİLDİ — ÖZET")
print("="*70)
print(f"\n{'Model':<20} {'AUC':>8} {'Accuracy':>10} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Süre(s)':>10}")
print("-" * 76)
for result in all_results:
    m = result["metrics"]
    print(f"{result['model_name']:<20} "
          f"{m['auc']:>8.4f} {m['accuracy']:>10.4f} {m['f1']:>8.4f} "
          f"{m['precision']:>10.4f} {m['recall']:>8.4f} "
          f"{result['train_time']:>10.1f}")

best_result = max(all_results, key=lambda r: r["metrics"]["auc"])
print(f"\n🏆 En İyi Model (AUC bazlı): {best_result['model_name']}")
print(f"   AUC: {best_result['metrics']['auc']:.4f}")

# ===============================================================
# 12. GÖRSELLEŞTİRME — Tüm görseller result/ klasörüne
# ===============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-darkgrid')

print("\n" + "="*70)
print("🎨 GÖRSELLEŞTİRMELER OLUŞTURULUYOR")
print("="*70)

results_data = []
for r in all_results:
    m = r["metrics"]
    results_data.append({
        "model_name": r["model_name"],
        "auc": float(m["auc"]), "accuracy": float(m["accuracy"]),
        "f1": float(m["f1"]), "precision": float(m["precision"]),
        "recall": float(m["recall"]),
        "tn": int(m["tn"]), "fp": int(m["fp"]),
        "fn": int(m["fn"]), "tp": int(m["tp"]),
        "train_time_seconds": float(r["train_time"]),
    })
results_pdf = pd.DataFrame(results_data).sort_values("auc", ascending=False).reset_index(drop=True)

colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

# GÖRSEL 1: 5 Modelin Performans Karşılaştırması
fig, ax = plt.subplots(figsize=(14, 7))
metrics_to_plot = ['auc', 'accuracy', 'f1', 'precision', 'recall']
metric_labels = ['AUC-ROC', 'Accuracy', 'F1-Score', 'Precision', 'Recall']
x = np.arange(len(results_pdf))
width = 0.15
for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
    offset = width * (i - len(metrics_to_plot)/2 + 0.5)
    bars = ax.bar(x + offset, results_pdf[metric], width,
                   label=label, color=colors[i], edgecolor='black', linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7)
ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Skor', fontsize=12, fontweight='bold')
ax.set_title('5 Modelin Performans Karşılaştırması', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(results_pdf['model_name'], rotation=15, ha='right')
ax.legend(loc='lower right', framealpha=0.9)
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(f"{RESULT_PATH}/01_model_performance_comparison.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 01_model_performance_comparison.png")

# GÖRSEL 2: Feature Importance
best_tree_result = None
for r in all_results:
    if r["feature_importances"] is not None:
        if best_tree_result is None or r["metrics"]["auc"] > best_tree_result["metrics"]["auc"]:
            best_tree_result = r

if best_tree_result is not None:
    importances = np.array(best_tree_result["feature_importances"])
    top_n = min(15, len(importances))
    top_idx = np.argsort(importances)[-top_n:]
    try:
        feature_names = train_df.schema["features"].metadata.get("ml_attr", {}).get("attrs", {})
        all_features = []
        for typ in ["numeric", "nominal", "binary"]:
            if typ in feature_names:
                all_features.extend([(a["idx"], a["name"]) for a in feature_names[typ]])
        all_features.sort()
        name_map = {idx: name for idx, name in all_features}
        feature_labels = [name_map.get(i, f"Feature_{i}") for i in top_idx]
    except Exception:
        feature_labels = [f"Feature_{i}" for i in top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(range(top_n), importances[top_idx],
                    color='#2E86AB', edgecolor='black', linewidth=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feature_labels, fontsize=10)
    ax.set_xlabel('Önem Skoru', fontsize=12, fontweight='bold')
    ax.set_title(f'Feature Importance — {best_tree_result["model_name"]} (Top {top_n})',
                 fontsize=14, fontweight='bold')
    for i, (bar, v) in enumerate(zip(bars, importances[top_idx])):
        ax.text(v + max(importances[top_idx])*0.01, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{RESULT_PATH}/02_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 02_feature_importance.png")

# GÖRSEL 3: Zaman Serisi Trend
time_col_candidates = ["issue_d", "issue_date", "earliest_cr_line", "date", "year", "issue_year"]
time_col = None
source_df = gold_df if gold_df is not None else train_df

for col in time_col_candidates:
    if col in source_df.columns:
        time_col = col
        break

if time_col is not None:
    try:
        time_data = (source_df.groupBy(time_col).count()
                     .orderBy(time_col).toPandas())
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(time_data[time_col].astype(str), time_data['count'],
                marker='o', color='#2E86AB', linewidth=2, markersize=6)
        ax.fill_between(range(len(time_data)), time_data['count'],
                         alpha=0.2, color='#2E86AB')
        ax.set_xlabel(time_col, fontsize=12, fontweight='bold')
        ax.set_ylabel('Kredi Sayısı', fontsize=12, fontweight='bold')
        ax.set_title(f'Zaman Serisi Trendi — {time_col} Bazında Kredi Sayısı',
                     fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{RESULT_PATH}/03_time_series_trend.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 03_time_series_trend.png (kolon: {time_col})")
    except Exception as e:
        print(f"⚠️  Zaman serisi grafiği oluşturulamadı: {e}")
else:
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_by_time = results_pdf.sort_values('train_time_seconds')
    ax.plot(sorted_by_time['model_name'], sorted_by_time['train_time_seconds'],
            marker='o', color='#C73E1D', linewidth=2, markersize=10)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Eğitim Süresi (saniye)', fontsize=12, fontweight='bold')
    ax.set_title('Modellerin Eğitim Süresi Trendi', fontsize=14, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f"{RESULT_PATH}/03_time_series_trend.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 03_time_series_trend.png (fallback)")

# GÖRSEL 4: Veri Dağılım
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
label_counts = (source_df.groupBy("label").count().orderBy("label").toPandas())
axes[0].pie(label_counts['count'],
            labels=[f'Class {int(l)} (Geri Ödenmiş)' if l == 0
                    else f'Class {int(l)} (Default)' for l in label_counts['label']],
            autopct='%1.1f%%', startangle=90,
            colors=['#2E86AB', '#C73E1D'],
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},
            textprops={'fontsize': 11, 'fontweight': 'bold'})
axes[0].set_title('Target (Default) Dağılımı', fontsize=13, fontweight='bold')

numeric_candidates = ["loan_amnt", "int_rate", "annual_inc", "dti", "fico_score"]
hist_col = None
for col in numeric_candidates:
    if col in source_df.columns:
        hist_col = col
        break

if hist_col is not None:
    sample = source_df.select(hist_col).sample(0.1).toPandas().dropna()
    q1, q99 = sample[hist_col].quantile([0.01, 0.99])
    sample = sample[(sample[hist_col] >= q1) & (sample[hist_col] <= q99)]
    axes[1].hist(sample[hist_col], bins=40, color='#F18F01',
                  edgecolor='black', linewidth=0.5)
    axes[1].set_xlabel(hist_col, fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frekans', fontsize=11, fontweight='bold')
    axes[1].set_title(f'{hist_col} Dağılımı (Histogram)', fontsize=13, fontweight='bold')
else:
    axes[1].text(0.5, 0.5, 'Sayısal kolon bulunamadı',
                 ha='center', va='center', fontsize=14)
    axes[1].axis('off')

plt.suptitle('Veri Dağılım Grafikleri', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{RESULT_PATH}/04_data_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 04_data_distribution.png")

# GÖRSEL 5: EDA Ek
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes[0, 0].bar(['Class 0\n(Geri Ödenmiş)', 'Class 1\n(Default)'],
                label_counts['count'],
                color=['#2E86AB', '#C73E1D'], edgecolor='black', linewidth=1)
for i, v in enumerate(label_counts['count']):
    axes[0, 0].text(i, v, f'{v:,}', ha='center', va='bottom',
                     fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Sayı', fontsize=11, fontweight='bold')
axes[0, 0].set_title('Sınıf Dağılımı (Class Imbalance)', fontsize=12, fontweight='bold')

axes[0, 1].bar(['Train', 'Test'], [train_count, test_count],
                color=['#2E86AB', '#A23B72'], edgecolor='black', linewidth=1)
for i, v in enumerate([train_count, test_count]):
    axes[0, 1].text(i, v, f'{v:,}', ha='center', va='bottom',
                     fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('Satır Sayısı', fontsize=11, fontweight='bold')
axes[0, 1].set_title('Train / Test Dağılımı', fontsize=12, fontweight='bold')

sorted_time = results_pdf.sort_values('train_time_seconds')
axes[1, 0].barh(sorted_time['model_name'], sorted_time['train_time_seconds'],
                 color=colors[:len(sorted_time)], edgecolor='black', linewidth=1)
for i, v in enumerate(sorted_time['train_time_seconds']):
    axes[1, 0].text(v, i, f' {v:.1f}s', va='center', fontsize=10, fontweight='bold')
axes[1, 0].set_xlabel('Saniye', fontsize=11, fontweight='bold')
axes[1, 0].set_title('Model Eğitim Süresi Karşılaştırması', fontsize=12, fontweight='bold')

for i, (_, row) in enumerate(results_pdf.iterrows()):
    axes[1, 1].scatter(row['train_time_seconds'], row['auc'],
                        s=250, color=colors[i % len(colors)],
                        edgecolor='black', linewidth=1.5, zorder=3,
                        label=row['model_name'])
    axes[1, 1].annotate(row['model_name'],
                        (row['train_time_seconds'], row['auc']),
                        xytext=(8, 5), textcoords='offset points', fontsize=9)
axes[1, 1].set_xlabel('Eğitim Süresi (s)', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('AUC-ROC', fontsize=11, fontweight='bold')
axes[1, 1].set_title('Performans vs Hız Trade-off', fontsize=12, fontweight='bold')
axes[1, 1].legend(loc='best', fontsize=8)

plt.suptitle('EDA Bulguları — Ek Görselleştirmeler', fontsize=15, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f"{RESULT_PATH}/05_eda_visualizations.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 05_eda_visualizations.png")

# GÖRSEL 6: Confusion Matrix
m = best_result["metrics"]
cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap='Blues', aspect='auto')
total = cm.sum()
cm_pct = cm / total * 100
for i in range(2):
    for j in range(2):
        text_color = 'white' if cm[i, j] > cm.max()/2 else 'black'
        ax.text(j, i, f'{cm[i, j]:,}\n({cm_pct[i,j]:.2f}%)',
                ha='center', va='center', color=text_color,
                fontsize=15, fontweight='bold')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Tahmin: 0\n(Geri Ödenmiş)', 'Tahmin: 1\n(Default)'],
                    fontsize=11, fontweight='bold')
ax.set_yticklabels(['Gerçek: 0\n(Geri Ödenmiş)', 'Gerçek: 1\n(Default)'],
                    fontsize=11, fontweight='bold')
ax.set_title(f'En İyi Modelin Confusion Matrix\'i\n'
             f'{best_result["model_name"]} (AUC={m["auc"]:.4f})',
             fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f"{RESULT_PATH}/06_best_model_confusion_matrix.png",
            dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 06_best_model_confusion_matrix.png")

# GÖRSEL 7: ROC Curve
from pyspark.ml.functions import vector_to_array

fig, ax = plt.subplots(figsize=(10, 8))
for i, result in enumerate(all_results):
    try:
        pred_df = result["predictions"]
        if "probability" not in pred_df.columns:
            print(f"   ⚠️  {result['model_name']}: probability yok, ROC atlandı")
            continue
        roc_data = (pred_df
                    .select(vector_to_array("probability").alias("prob"), "label")
                    .selectExpr("prob[1] as score", "label")
                    .sample(0.2, seed=42)
                    .toPandas())
        from sklearn.metrics import roc_curve, auc as sk_auc
        fpr, tpr, _ = roc_curve(roc_data['label'], roc_data['score'])
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                label=f"{result['model_name']} (AUC={roc_auc:.4f})")
    except Exception as e:
        print(f"   ⚠️  {result['model_name']} ROC çizilemedi: {e}")

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve — Tüm Modeller', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig(f"{RESULT_PATH}/07_roc_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 07_roc_curve.png")

# ---------------------------------------------------------------
# 13. MLflow'a görselleri de artifact olarak yükle
# ---------------------------------------------------------------
print("\n📤 Görseller MLflow'a artifact olarak yükleniyor...")
try:
    with mlflow.start_run(run_name="All_Visualizations"):
        for fname in sorted(os.listdir(RESULT_PATH)):
            if fname.endswith(".png"):
                mlflow.log_artifact(os.path.join(RESULT_PATH, fname),
                                     artifact_path="plots")
        mlflow.log_param("best_model", best_result["model_name"])
        mlflow.log_metric("best_auc", best_result["metrics"]["auc"])
    print("✅ Görseller MLflow'a yüklendi")
except Exception as e:
    print(f"⚠️  MLflow artifact yükleme hatası: {e}")

# ---------------------------------------------------------------
# 14. Final Özet
# ---------------------------------------------------------------
print("\n" + "="*70)
print("🎉 TÜM İŞLEMLER TAMAMLANDI")
print("="*70)
print(f"""
📂 Görseller: {RESULT_PATH}
   1. 01_model_performance_comparison.png
   2. 02_feature_importance.png
   3. 03_time_series_trend.png
   4. 04_data_distribution.png
   5. 05_eda_visualizations.png
   6. 06_best_model_confusion_matrix.png
   7. 07_roc_curve.png

📈 MLflow tracking:
   URI: {MLFLOW_TRACKING_URI}
   Experiment: {EXPERIMENT_NAME}

🏆 En İyi Model: {best_result['model_name']}
   AUC:       {best_result['metrics']['auc']:.4f}
   Accuracy:  {best_result['metrics']['accuracy']:.4f}
   F1:        {best_result['metrics']['f1']:.4f}
""")

spark.stop()
print("✅ Spark session kapatıldı")