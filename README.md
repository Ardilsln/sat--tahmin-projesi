# 📊 Satış Tahmini API Projesi

Bu proje, Northwind veritabanı kullanılarak geçmiş satış verileriyle ürün bazlı satış tahminleri yapan bir makine öğrenmesi modelini içerir. Model, FastAPI ile geliştirilmiş REST API aracılığıyla dış sistemlerin tahmin talebinde bulunmasını sağlar.

## ✨ Temel Özellikler

- PostgreSQL veritabanından veri çekme
- Verileri temizleme, birleştirme ve öznitelik çıkarma
- Decision Tree Regressor ile model eğitimi
- REST API ile satış tahmini servis etme
- FastAPI Swagger arayüzüyle test edilebilir endpoint'ler

## 🔧 Kurulum

### 1. Bağımlılıkları Kur
```bash
pip install fastapi uvicorn pandas scikit-learn sqlalchemy psycopg2-binary joblib
```

### 2. PostgreSQL Ayarları
`db.py` içindeki aşağıdaki bilgileri kendi veritabanına göre düzenle:
```python
params = {
  'database': 'GYK2Northwind',
  'user': 'postgres',
  'password': 'gykproject2025',
  'host': 'localhost',
  'port': '5432'
}
```

### 3. Veri Hazırlığı & Model Eğitimi
```bash
python db.py
```
Bu adım sonunda:
- `features.csv` üretilir
- `sales_forecast_model.pkl` dosyası modele kaydedilir

### 4. API Sunucusunu Başlat
```bash
python -m uvicorn main:app --reload
```
Ardından tarayıcıdan aşağıdaki adrese giderek Swagger arayüzüne ulaşabilirsiniz:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🌐 API Referans Dökümantasyonu

### ✉️ `POST /predict`
Verilen ürün ve sipariş bilgilerine göre tahmini miktar döner.

#### İstek (JSON):
```json
{
  "month": "1997-07",
  "category_name": "Beverages",
  "country": "Germany",
  "customer_desc": "Unknown",
  "unit_price_x": 18.0,
  "product_name": "Chai",
  "total_sales": 180.0
}
```

#### Yanıt:
```json
{
    "predicted_quantity": 10.0,
    "product_name": "Chai"
}
```

### 📅 `GET /products`
Tüm benzersiz ürün isimlerini listeler.

### 📊 `GET /sales_summary`
Tüm ürünler için toplam satış tutarını döner.

### 🎓 `POST /retrain`
Modeli mevcut `features.csv` verisiyle yeniden eğitir ve günceller.

---

## 📁 Proje Yapısı
```
BIG_PROJECT_PAIR4/
├── db.py                  # Veri çekme ve model eğitimi
├── main.py                # FastAPI uygulaması
├── features.csv           # Eğitim verisi
├── sales_forecast_model.pkl # Kaydedilmiş model
```

---

## 📅 Katkıda Bulunanlar

- GYK Pair 4 Ekibi
* Nazlıcan ÇELİK
* Ardıl Silan AYDIN
* Sinem TAŞKIN
* Elif SİVRİ
* Emine GÜZELÖZ
- Turkcell Geleceği Yazan Kadınlar 2025

---

## 🔐 Notlar

- Bu proje sadece local geliştirme içindir.
- Gerçek dünyada kullanılacaksa sunucuya deploy edilmelidir.
- Swagger arayüzü sayesinde API kolayca test edilebilir.
