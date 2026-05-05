# Kredi Risk Analizi: Uçtan Uca Büyük Veri Boru Hattı (Data Pipeline)

Bu proje, Lending Club kredi veri seti kullanılarak geliştirilen uçtan uca bir büyük veri mimarisidir. Projenin bu ilk fazında, devasa boyutlardaki ham veriler Kafka ile anlık olarak akıtılmış, Apache Spark ile işlenmiş ve analitik süreçler (Makine Öğrenmesi & Dashboard) için Delta/Parquet Lakehouse formatında sıkıştırılarak kaydedilmiştir.

## 🏗️ Kullanılan Teknolojiler
* **Apache Kafka & Zookeeper:** Gerçek zamanlı veri akışı (Data Streaming)
* **Apache Spark (PySpark):** Büyük veri işleme (Structured Streaming)
* **Docker & Docker Compose:** İzole ve taşınabilir çalışma ortamı
* **Parquet:** Sıkıştırılmış sütunsal veri depolama formatı

## 🚀 Sistemi Yerelde Çalıştırma Adımları

**1. Altyapıyı Ayağa Kaldırma**
Projeyi indirdikten sonra terminal üzerinden Docker ortamını başlatın:
```bash
docker-compose up -d