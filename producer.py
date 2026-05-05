import json
import csv
import time
from kafka import KafkaProducer

# Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',  # host'tan bağlandığın için doğru
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    linger_ms=10,            # küçük batching (performans artışı)
    batch_size=16384,
    acks='all',              # veri güvenliği
    retries=5
)

topic_name = 'lending_club_stream'
csv_file_path = 'accepted_2007_to_2018Q4.csv'

print(f"Veri akışı '{topic_name}' topic'ine başlıyor...")

with open(csv_file_path, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    for count, row in enumerate(reader, start=1):
        try:
            producer.send(topic_name, value=row)

            if count % 1000 == 0:
                print(f"{count} kayıt gönderildi...")
                time.sleep(0.1)


        except Exception as e:
            print(f"Hata oluştu: {e}")

# buffer flush
producer.flush()
producer.close()

print("Tüm veri başarıyla Kafka'ya gönderildi!")