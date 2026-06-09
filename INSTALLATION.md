# Kurulum Kılavuzu

## Gereksinimler

| Gereksinim | Versiyon |
|------------|---------|
| Python | 3.9 |
| pip | ≥23.0 |
| ffmpeg | herhangi |
| RAM | ≥8 GB (eğitim için ≥16 GB) |
| Disk | ≥3 GB (MP3 + cache) |

> GPU isteğe bağlıdır. CPU ile de çalışır ancak eğitim çok yavaşlar.

---

## 1. Depoyu Klonla

```bash
git clone https://github.com/yusuf7855/trackbang-key-detection.git
cd trackbang-key-detection
```

---

## 2. Sanal Ortam Oluştur

```bash
python3.9 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

---

## 3. Bağımlılıkları Kur

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> İlk kurulum ~5 dakika sürebilir (TensorFlow boyutu nedeniyle).

### ffmpeg Kurulumu

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
# https://ffmpeg.org/download.html adresinden indirip PATH'e ekle
```

---

## 4. Kurulumu Doğrula

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import librosa; print(librosa.__version__)"
python src/model.py   # model mimarisini yazdırır
```

Beklenen çıktı:
```
2.12.x (veya üzeri)
0.10.x (veya üzeri)
Model: "key_bpm_model"
Total params: 858,009
```

---

## 5. Modeli İndir

Eğitilmiş v4 model dosyası Git LFS ile takip edilmektedir:

```bash
git lfs pull
# veya doğrudan Releases sayfasından indir:
# https://github.com/yusuf7855/trackbang-key-detection/releases
```

Model dosyası: `models/key_detection_v4_best.keras`

---

## 6. Hızlı Test — Tek Dosya Tahmini

```bash
python src/predict.py data/raw/previews/<herhangi_bir>.mp3
```

Beklenen çıktı:
```json
{
  "key": "A Minor",
  "camelot": "8A",
  "confidence": 0.41,
  "bpm": 127.4,
  "top3": [
    {"key": "A Minor", "camelot": "8A", "confidence": 0.41},
    {"key": "E Minor", "camelot": "9A", "confidence": 0.09},
    {"key": "D Minor", "camelot": "7A", "confidence": 0.06}
  ]
}
```

---

## 7. API Sunucusunu Başlat

```bash
python src/api.py
# veya
uvicorn src.api:app --reload --port 8000
```

Tarayıcıdan test: `http://localhost:8000`  
Swagger UI: `http://localhost:8000/docs`

```bash
# curl ile test
curl -X POST http://localhost:8000/predict \
     -F "file=@ornek_sarki.mp3"
```

---

## 8. Veri Setiyle Eğitim (İsteğe Bağlı)

> ⚠️ Bu adım için MongoDB erişimi gerekmektedir.

```bash
# Önce veriyi indir
python src/build_dataset.py --limit 100   # test için
python src/build_dataset.py               # tamamı (3.817 şarkı)

# Eğitimi başlat
python src/train.py --epochs 120 --batch 16

# feature_extraction.py değiştiyse cache'i yenile
python src/train.py --recache --epochs 120
```

Eğitim çıktısı `results/training_log.csv` dosyasına kaydedilir.

---

## Sık Karşılaşılan Sorunlar

### `librosa` yüklenmiyor
```bash
pip install soundfile
```

### TensorFlow GPU tanımıyor
```bash
pip install tensorflow[and-cuda]   # CUDA 12+ için
```

### `ffmpeg` bulunamıyor
`ffmpeg`'in PATH'te olduğunu doğrula:
```bash
ffmpeg -version
```

### Bellek hatası (OOM)
`src/train.py` içinde batch boyutunu küçült:
```bash
python src/train.py --batch 8
```
