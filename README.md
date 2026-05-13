# 🚀 Kredi Risk Analizi: Uçtan Uca Büyük Veri ve Makine Öğrenmesi Boru Hattı

Bu proje, Lending Club kredi veri seti kullanılarak geliştirilen uçtan uca bir Büyük Veri (Big Data) mimarisidir. Proje kapsamında devasa boyutlardaki ham veriler **Apache Kafka** ile gerçek zamanlı olarak akıtılmış, **Apache Spark** ile işlenerek Medallion mimarisinde (Silver ve Gold katmanlar) Delta Lake formatında saklanmıştır. Son aşamada ise temizlenen veriler üzerinde **Spark MLlib** ile 5 farklı makine öğrenmesi modeli eğitilmiş, sonuçlar **MLflow** ile loglanmış ve detaylı görselleştirmeler sunulmuştur.

## 🏗️ Kullanılan Teknolojiler ve Mimari
* **Veri Akışı (Streaming):** Apache Kafka & Zookeeper
* **Büyük Veri İşleme:** Apache Spark (PySpark), Structured Streaming
* **Veri Depolama (Lakehouse):** Delta Lake, Parquet
* **Makine Öğrenmesi:** Spark MLlib (Logistic Regression, LinearSVC, Decision Tree, Random Forest, GBT)
* **Deney Takibi (MLOps):** MLflow
* **Görselleştirme & EDA:** Matplotlib, Seaborn, Pandas
* **Altyapı:** Docker & Docker Compose (özelleştirilmiş `Dockerfile.spark` ile)

## 🚀 Sistemi Yerelde Çalıştırma Adımları

**1. Altyapıyı Ayağa Kaldırma**

Projeyi GitHub'dan indirdikten sonra, orijinal Kaggle verilerini (`accepted_...csv` vb.) ana dizine kopyalayın ve terminal üzerinden Docker ortamını başlatın:

```bash
docker-compose up -d
```

Bu komut Kafka, Zookeeper ve içinde gerekli tüm MLOps kütüphanelerinin (MLflow vb.) bulunduğu özelleştirilmiş Spark motorunu kuracaktır.

**2. Canlı Veri Akışını Başlatma (Producer)**

Kafka'ya verileri canlı bir sistemden geliyormuş gibi pompalamak için yeni bir terminalde şu betiği çalıştırın:

```bash
python producer.py
```

**3. Uçtan Uca Analitik Boru Hattını Çalıştırma (Otomasyon)**

Sistem ayağa kalktıktan sonra, verinin gümüş katmandan altın katmana geçişini, EDA analizlerini, model eğitimini ve görselleştirmelerin oluşturulmasını tek bir komutla (veya `localhost:8888` Jupyter arayüzünden hücre hücre) tetikleyebilirsiniz.

Terminalden otomatik çalıştırmak için:

```bash
docker exec -it spark-jupyter bash -c "set -e; cd /home/jovyan/work; \
echo '### ADIM 1: CSV -> Silver ###'; spark-submit --packages io.delta:delta-spark_2.12:3.0.0 consumer/backfill_csv_to_silver.py; \
echo '### ADIM 2: EDA Analizi ###'; spark-submit --packages io.delta:delta-spark_2.12:3.0.0 notebooks/gold_preprocessing.py; \
echo '### ADIM 3: Silver -> Gold + Train/Test Split ###'; spark-submit --packages io.delta:delta-spark_2.12:3.0.0 notebooks/eda_analysis.py && spark-submit --packages io.delta:delta-spark_2.12:3.0.0 notebooks/model_dataset_preparation.py; \
echo '### ADIM 4: Model Eğitimi, MLflow ve Görselleştirme ###'; spark-submit --packages io.delta:delta-spark_2.12:3.0.0 notebooks/model_training.py; \
echo '### TÜM SÜREÇ BAŞARIYLA TAMAMLANDI ###'"
```

## 📈 Çıktılar ve Sonuçlar

Tüm modellerin çalışması tamamlandığında:

- **En iyi model:** `GBTClassifier` (Gradient-Boosted Trees) ~0.738 AUC skoru ile en iyi performansı göstermiştir.
- **Deney logları:** Eğitim süreleri, parametreler ve metrikler MLflow sunucusuna kaydedilmiştir (Jupyter içinden `mlruns` dizini altından incelenebilir).
- **Görseller:** Hocaya sunulacak tüm grafikler (performans karşılaştırmaları, ROC eğrisi, zaman serisi, confusion matrix) otomatik olarak `result/` klasörünün içine yüksek çözünürlüklü `.png` dosyaları olarak kaydedilmiştir.