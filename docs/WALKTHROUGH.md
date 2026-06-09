# Proje Yürüyüşü (Walkthrough)

Bu belge, projeyi ilk kez inceleyen bir geliştirici veya araştırmacı için adım adım rehberdir.

---

## 1. Projeyi Anlamak — 10 Dakika

### Neyi Çözüyor?
Bir MP3 dosyası ver → müzikal tonu (örn. "A Minor / 8A") ve BPM'i al.

### Neden Önemli?
TrackBang'in CUE AI asistanı, "8A tonunda 127 BPM peaktime set" gibi istekleri karşılamak için şarkıların key/BPM bilgisine ihtiyaç duyar. Bu model o bilgiyi üretir.

### Hızlı Kavramsal Harita
```
MP3 (30sn)
  └─ HPSS → harmonik bileşen
       └─ Chroma CENS → (12, 1292) matris
            └─ Hibrit CNN (v4)
                 ├─ key_output → "A Minor" (24 sınıf, softmax)
                 └─ bpm_output → 127.4 BPM (regresyon)
```

---

## 2. Repo Yapısını Keşfet — 5 Dakika

```
trackbang-key-detection/
│
├── src/                   ← Tüm Python kodu burada
│   ├── model.py           ← CNN mimarisi (buradan başla)
│   ├── feature_extraction.py  ← MP3 → Chroma matris
│   ├── train.py           ← Eğitim döngüsü
│   ├── predict.py         ← Tek dosya tahmini
│   └── api.py             ← FastAPI REST servisi
│
├── data/
│   ├── labels.csv         ← 3.817 şarkı etiketi
│   └── raw/previews/      ← MP3 dosyaları
│
├── models/
│   └── key_detection_v4_best.keras  ← Kullanıma hazır model
│
├── results/
│   ├── training_log.csv   ← Eğitim metrikleri
│   └── plots/             ← Eğitim grafikleri
│
├── docs/                  ← Belgeler
├── future_work/           ← Gelecek planlar
│
├── AGENTS.md              ← Detaylı teknik referans (buradan devam et)
├── INSTALLATION.md        ← Kurulum adımları
└── CHANGELOG.md           ← Versiyon geçmişi
```

---

## 3. Kodu Oku — 20 Dakika

### Adım 1: Model Mimarisini Anla
```bash
cat src/model.py
```
`build_model()` fonksiyonu tüm mimariyi döndürür. Residual bloklar ve SE attention bloklarına dikkat et.

### Adım 2: Özellik Çıkarmayı Anla
```bash
cat src/feature_extraction.py
```
`extract_all_features(file_path)` → `(12, 1292)` numpy array döndürür. HPSS adımı neden gerekli, yorumlara bak.

### Adım 3: Eğitim Pipeline'ını Anla
```bash
cat src/train.py
```
`train_val_split()` → veri bölünme stratejisi (DB verisi vs SoundCloud verisi ayrımı kritik).

### Adım 4: API'yi Anla
```bash
cat src/api.py
```
`POST /predict` endpoint'i: multipart/form-data MP3 alır, JSON döndürür.

---

## 4. Çalıştır — 15 Dakika

### Tahmin Yap (en hızlı yol)
```bash
source venv/bin/activate
python src/predict.py data/raw/previews/<herhangi>.mp3
```

### API'yi Başlat
```bash
python src/api.py
# http://localhost:8000/docs adresine git
```

### Eğitim Metriklerini İncele
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/training_log.csv')
df['val_key_output_accuracy'].plot(title='Validation Key Accuracy')
plt.show()
```

---

## 5. Veri Setini Anla — 10 Dakika

```python
import pandas as pd
df = pd.read_csv('data/labels.csv')

print(df.shape)                          # (3817, 6)
print(df['musical_key'].value_counts())  # sınıf dağılımı
print(df['source'].value_counts())       # db vs soundcloud
```

**Önemli:** `source == 'db'` satırları yüksek kaliteli etiket (DJ'ler tarafından girilmiş). `source == 'sc'` satırları SoundCloud metadata'sından gelir, gürültülü olabilir.

---

## 6. Versiyon Geçmişini Anla

| Versiyon | Mimari | Val Acc | Not |
|----------|--------|---------|-----|
| v1 | Temel 3-blok CNN | %35.3 | İlk çalışan model |
| v2 | Residual + SE Attention | ~%40 | HPSS eklendi |
| v3 | Dual-branch (Chroma + Mel) | — | Başarısız, overfitting |
| v4 | Hibrit GAP+GMP + TTA | %47.7 | **Güncel model** |

Detaylar için: [CHANGELOG.md](../CHANGELOG.md)

---

## 7. Sisteme Katkı Vermek

1. Fork yap
2. `feature/açıklama` branch'i aç
3. Değişikliğini yap, test et
4. `CHANGELOG.md`'yi güncelle
5. Pull Request aç

Açık sorunlar ve fikirler için: [future_work/TODO.md](../future_work/TODO.md)
