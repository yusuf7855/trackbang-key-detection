# AGENTS.md — TrackBang Key Detection: Teknik Referans Kılavuzu

> Bu dosya, projeyle ilgili çalışacak her türlü AI agent, geliştirici veya araştırmacı için
> kapsamlı teknik bağlam sağlar. Projeyi anlamak için buradan başlayın.

---

## 1. Projenin Amacı ve Bağlamı

### Ne Yapıyor?
30 saniyelik MP3 ses dosyasından iki şeyi tahmin eder:
1. **Müzikal ton (Key)** — 24 sınıf (12 Majör + 12 Minör), Camelot notasyonuyla
2. **BPM** — sürekli sayısal değer (regresyon)

### Neden Önemli?
- TrackBang, iOS/Android'de yayında olan bir DJ müzik keşif platformu
- CUE adlı AI DJ asistanı, Groq LLM kullanarak kullanıcının doğal dil isteğini analiz eder
- "8A tonunda karanlık peaktime set" gibi istekleri karşılamak için şarkıların key/BPM bilgisi gerekli
- Bu model, o bilgileri otomatik olarak üretir (3rd party API bağımlılığı yok)

### Büyük Resim: TrackBang Mimarisi
```
[Mobil App: Flutter]
        │ HTTP
        ▼
[Backend: Node.js + Express]  ←→  [MongoDB + Redis]
        │
        ├── CUE AI (cueRoutes.js) — Groq LLM intent parsing
        │        └── Müzik öneri motoru (Camelot uyumluluk, BPM eşleştirme)
        │
        └── Key/BPM API (bu proje) — FastAPI
                 └── TensorFlow model (key_detection_model.keras)
```

---

## 2. Proje Dosya Haritası

```
trackbang-key-detection/
├── src/
│   ├── model.py              — CNN mimarisi + GAP/GMP paralel dallar (build_model fonksiyonu)
│   ├── feature_extraction.py — MP3 → Chroma CENS dönüşümü (HPSS dahil)
│   ├── train.py              — Eğitim loop'u (tf.data, augment, class weights)
│   ├── predict.py            — Inference + TTA: predict(mp3_path) → dict
│   ├── api.py                — FastAPI: POST /predict, GET /health
│   ├── build_dataset.py      — Veri toplama: MongoDB + SoundCloud → MP3 + labels.csv
│   ├── scraper.py            — Ek veri scraper
│   └── monitor.py            — Eğitim dashboard (web UI, port 8765)
│
├── data/
│   ├── labels.csv            — 3.817 satır: song_id, filename, musical_key, bpm, camelot, energy
│   ├── raw/previews/         — 2.191 MP3 dosyası (30 sn, ~1.1 GB toplam)
│   └── processed/cache/      — 3.817 .npy dosyası (12×1292 float32 Chroma CENS)
│
├── models/
│   └── key_detection_v4_best.keras  — v4 Hibrit CNN, aktif model (858K parametre)
│
├── results/
│   ├── training_log.csv      — epoch, loss, accuracy, mae, lr kolonları
│   └── plots/                — Eğitim grafikleri (training_curves.png)
│
├── docs/
│   ├── idea.md               — Proje fikri ve motivasyon
│   ├── proje_metni.md        — Resmi proje metni (öğrenci/danışman bilgisi)
│   ├── WALKTHROUGH.md        — Adım adım proje rehberi (yeni gelenlere)
│   └── screenshots/          — Uygulama ekran görüntüleri
│
├── future_work/
│   └── TODO.md               — Gelecek çalışmalar, bilinen hatalar, katkı rehberi
│
├── AGENTS.md                 — Bu dosya (teknik referans)
├── README.md                 — Kullanıcı/geliştirici belgesi
├── CHANGELOG.md              — Versiyon geçmişi (v1 → v4)
├── INSTALLATION.md           — Kurulum, çalıştırma ve troubleshooting rehberi
├── requirements.txt          — Python bağımlılıkları
└── generate_report.py        — python-docx ile Word raporu üretir
```

---

## 3. Model Mimarisi (v4 — Güncel)

### Dosya: `src/model.py`

**Ana fonksiyon:** `build_model(input_shape=(12, 1292, 1), lr=0.001) → tf.keras.Model`

**Giriş:** `(batch, 12, 1292, 1)` — 12 chroma bin × 1292 zaman adımı × 1 kanal

**Çıktılar:**
- `key_output`: `(batch, 24)` — softmax olasılıkları
- `bpm_output`: `(batch, 1)` — normalize BPM (bpm/200.0)

### Katman Akışı (v4 — Hibrit GAP+GMP)

```python
Input(12, 1292, 1)
  → Conv2D(64, 3×7) + BN + LeakyReLU(0.1)
  → ResidualBlock(64, 3×7) + SE_Block(ratio=8)
  → MaxPooling2D(1×4)   # zaman: 1292 → 323
  → Dropout(0.2)
  → ResidualBlock(128, 3×5) + SE_Block(ratio=8)
  → MaxPooling2D(1×4)   # zaman: 323 → 80
  → Dropout(0.2)
  → ResidualBlock(256, 3×3) + SE_Block(ratio=8)
  → MaxPooling2D(1×4)   # zaman: 80 → 20
  → Dropout(0.2)
  → ResidualBlock(256, 3×3) + SE_Block(ratio=8)
  → Dropout(0.2)
  ├→ GlobalAveragePooling2D()   # (batch, 256) — ortalama aktivasyon
  └→ GlobalMaxPooling2D()       # (batch, 256) — zirve aktivasyon
  → Concatenate()               # (batch, 512)
  → Dense(256) + BN + LeakyReLU + Dropout(0.4)
  ├→ Dense(24, softmax)   [key_output]
  └→ Dense(1, linear)     [bpm_output]
```

**TTA (Test-Time Augmentation):** Tahmin sırasında aynı parça 5 farklı augmentasyonla işlenir, softmax olasılıkları ortalaması alınır. Bu ~2–3% val accuracy artışı sağlar.

### SE Block (Squeeze-and-Excitation)

```python
def se_block(x, ratio=8):
    channels = x.shape[-1]
    se = GlobalAveragePooling2D()(x)         # (B, C)
    se = Reshape((1, 1, channels))(se)        # (B, 1, 1, C)
    se = Dense(channels // ratio, relu)(se)   # bottleneck
    se = Dense(channels, sigmoid)(se)         # kanal ağırlıkları
    return Multiply()([x, se])               # ağırlıklı özellikler
```

### Residual Block

```python
def residual_block(x, filters, kernel_size):
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNorm()(x)
    x = LeakyReLU(0.1)(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNorm()(x)
    x = se_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, 1×1)(shortcut)  # boyut eşleştirme
        shortcut = BatchNorm()(shortcut)
    x = Add()([x, shortcut])
    return LeakyReLU(0.1)(x)
```

### Kayıp Fonksiyonu

```
Total Loss = 1.0 × CategoricalCrossentropy(key, label_smoothing=0.1)
           + 0.05 × MSE(bpm)
```

Key accuracy'ye odaklanmak için BPM ağırlığı düşük tutulmuştur (0.05).

### Parametre Sayısı
- **Toplam:** 858.009 parametre
- **Eğitilebilir:** 858.009 parametre
- **Model boyutu:** ~3.5 MB (TF native Keras)
- **Model dosyası:** `models/key_detection_v4_best.keras`

### KEY_CLASSES Sıralaması
```python
KEY_CLASSES = [
    'C Major', 'C Minor',       # index 0, 1
    'C# Major', 'C# Minor',     # index 2, 3
    'D Major', 'D Minor',       # index 4, 5
    'D# Major', 'D# Minor',     # index 6, 7
    'E Major', 'E Minor',       # index 8, 9
    'F Major', 'F Minor',       # index 10, 11
    'F# Major', 'F# Minor',     # index 12, 13
    'G Major', 'G Minor',       # index 14, 15
    'G# Major', 'G# Minor',     # index 16, 17
    'A Major', 'A Minor',       # index 18, 19
    'A# Major', 'A# Minor',     # index 20, 21
    'B Major', 'B Minor',       # index 22, 23
]
```
**Önemli:** Her çift (Majör, Minör) aynı kroma notasına karşılık gelir.
Pitch shift augmentasyonunda `semitone * 2` formülü bu düzenin gereğidir:
- 1 semitone = chroma'da 1 satır kaydırma = index'te 2 adım

---

## 4. Özellik Çıkarma (Feature Extraction)

### Dosya: `src/feature_extraction.py`

**Ana fonksiyon:** `extract_all_features(file_path, target_length=1292) → np.ndarray (12, 1292) float32`

### Pipeline Adımları

```
MP3 dosyası
  │
  ├─ librosa.load(sr=22050, duration=30, mono=True)
  │  → y: np.ndarray (660.150,)
  │
  ├─ librosa.effects.hpss(y, margin=3.0)
  │  → y_harmonic: perküsif sesler temizlendi
  │
  ├─ librosa.feature.chroma_cens(y=y_harmonic, sr=22050, hop_length=512, n_chroma=12)
  │  → chroma: (12, T)  T≈1292 frame
  │
  ├─ Uzunluk standardizasyonu
  │  if T >= 1292: kırp
  │  if T < 1292:  sıfır doldur
  │  → (12, 1292)
  │
  └─ Z-score normalizasyon: (x - mean) / (std + 1e-8)
     → float32 (12, 1292)
```

### Sabitler
```python
SAMPLE_RATE = 22050   # Hz
DURATION    = 30      # saniye
HOP_LENGTH  = 512     # ~23ms per frame
N_CHROMA    = 12      # pitch class sayısı
TARGET_LEN  = 1292    # 30sn @ 22050Hz / 512 = 1292.38 frame
```

### HPSS Neden Önemli?
- Perküsif sesler (kick, snare, hi-hat) chroma matrisine gürültü katar
- Harmonik bileşen = akorlar + melodiler = key bilgisi
- `margin=3.0`: agresif ayrım (daha temiz harmonik bileşen)
- Bu tek iyileştirme ~+5-10% accuracy artışı sağlar

### Cache Sistemi
- Her MP3 işlendikten sonra `.npy` olarak `data/processed/cache/` altına kaydedilir
- Eğitimde `tf.py_function` ile anında yüklenir, RAM'e sığar
- `feature_extraction.py` değiştiğinde: `python src/train.py --recache`

---

## 5. Eğitim Pipeline

### Dosya: `src/train.py`

**Çalıştırma:**
```bash
python src/train.py                            # varsayılan (120 epoch, batch=16)
python src/train.py --recache                  # feature extraction değişince
python src/train.py --epochs 200 --batch 32    # özel parametreler
python src/train.py --lr 0.0005                # daha düşük öğrenme hızı
```

### Veri Bölünmesi Stratejisi

```
labels.csv (3.817 örnek)
    │
    ├── db_rows   (MongoDB kaynaklı, yüksek kalite etiket)
    │     ├── train: %85 — stratified split
    │     └── val:   %15 — stratified split  ← SADECE bu validation'da
    │
    └── sc_rows   (SoundCloud scraper, gürültülü etiket)
          └── train: %100 ← validation'a GİRMEZ
```

Validation seti yalnızca temiz etiketli DB verisinden oluşur.
Bu, gerçekçi doğrulama skoru elde etmek için kritiktir.

### Augmentasyon (eğitim setine, on-the-fly)

```python
def augment_fn(chroma, labels):
    # 1. Pitch shift: ±3 semitone
    semitone = tf.random.uniform([], -3, 4, tf.int32)
    chroma   = tf.roll(chroma, semitone, axis=0)
    # Etiket güncelle: 1 semitone = 2 index
    new_key_idx = (key_idx + semitone * 2) % 24

    # 2. Time masking: 0-100 frame sıfırla
    t_width = tf.random.uniform([], 0, 100, tf.int32)
    t_start = tf.random.uniform([], 0, 1292 - t_width, tf.int32)
    # [t_start, t_start+t_width) aralığı sıfırlanır

    # 3. Frequency masking: 0-3 chroma bin sıfırla
    f_width = tf.random.uniform([], 0, 3, tf.int32)
    f_start = tf.random.uniform([], 0, 12 - f_width, tf.int32)

    # 4. Gaussian gürültü
    noise = tf.random.normal(shape, 0.0, 0.02)
    return chroma + noise
```

### Sınıf Ağırlıkları (Class Weights)

```python
# Inverse-frequency ağırlıklandırma
weight_i = N_total / (N_classes * count_i)
# Max ağırlık 5.0 ile sınırlandırılır (instability önleme)
```

Nadir sınıflar (C# Minör: 83 örnek) yüksek ağırlık alır,
bol sınıflar (G Minör: 275 örnek) düşük ağırlık alır.

### Öğrenme Hızı: Cosine Decay

```python
lr_schedule = CosineDecay(
    initial_learning_rate=0.001,
    decay_steps=total_steps,    # len(train) / batch * epochs
    alpha=1e-6                  # minimum LR
)
```

### Callback'ler

```python
ModelCheckpoint(monitor="val_key_output_accuracy", save_best_only=True)
EarlyStopping(monitor="val_key_output_accuracy", patience=20)
CSVLogger("results/training_log.csv")
```

### Beklenen Eğitim Çıktısı (v4)

```
Epoch 1/120
... - key_output_accuracy: 0.06 - val_key_output_accuracy: 0.05
...
Epoch 55/120
... - key_output_accuracy: 0.45 - val_key_output_accuracy: 0.47
...
Epoch 75/120 — EarlyStopping triggered (patience=20)
Best val_key_output_accuracy: 0.477
```

---

## 6. Inference / Tahmin

### Dosya: `src/predict.py`

**Ana fonksiyon:**
```python
predict(mp3_path: str, model_path: str = DEFAULT_MODEL) → dict
```

**Dönüş değeri:**
```python
{
    "key":        "A Minor",     # tahmin edilen ton
    "camelot":    "8A",          # Camelot kodu
    "confidence": 0.412,         # softmax olasılığı [0-1]
    "bpm":        127.4,         # bpm_output * 200
    "top3": [
        ("A Minor", "8A", 0.412),
        ("E Minor", "9A", 0.089),
        ("D Minor", "7A", 0.064),
    ]
}
```

**BPM denormalizasyon:**
```python
bpm_val = float(bpm_norm[0][0]) * 200.0     # model 0-1 çıkarır, 200 ile çarp
bpm_val = max(60.0, min(220.0, bpm_val))    # sınırla
```

**Camelot → Key Eşleme:**
```python
KEY_TO_CAMELOT = {
    'C Major': '8B',  'C Minor': '5A',
    'C# Major': '3B', 'C# Minor': '12A',
    'D Major': '10B', 'D Minor': '7A',
    # ... 24 çift
}
```

---

## 7. REST API

### Dosya: `src/api.py`

**Çalıştırma:**
```bash
python src/api.py
# veya
uvicorn src.api:app --reload --port 8000
```

**Endpoints:**

| Method | Path | Açıklama |
|--------|------|----------|
| GET | `/` | HTML test arayüzü |
| GET | `/health` | Model yüklü mü kontrolü |
| POST | `/predict` | MP3 yükle, key+BPM al |
| GET | `/docs` | Swagger UI |

**POST /predict:**
```bash
curl -X POST http://localhost:8000/predict -F "file=@track.mp3"
```

Response şeması:
```json
{
  "key":        "string",
  "camelot":    "string",
  "confidence": "float [0-1]",
  "bpm":        "float",
  "top3": [{"key":"string","camelot":"string","confidence":"float"}]
}
```

**CORS:** Tüm originlere açık (`allow_origins=["*"]`)

---

## 8. Veri Toplama

### Dosya: `src/build_dataset.py`

**Çalıştırma:**
```bash
python src/build_dataset.py              # hepsini indir
python src/build_dataset.py --limit 100  # test için ilk 100
python src/build_dataset.py --workers 3  # paralel indirme
```

**Pipeline:**
1. SSH tüneli → TrackBang MongoDB (root@72.62.63.184)
2. `platformLinks.soundcloud` + geçerli `musicalKey` filtresi
3. `yt-dlp` ile SoundCloud'dan MP3 indir
4. `ffmpeg` ile 0–30 saniye kırp
5. `data/raw/previews/<song_id>.mp3` kaydet
6. `data/labels.csv` güncelle

**labels.csv kolonları:**
```
song_id | filename | musical_key | bpm | camelot | energy
```

---

## 9. Eğitim Sonuçları

### results/training_log.csv Formatı
```
epoch, bpm_output_loss, bpm_output_mae, key_output_accuracy,
key_output_loss, learning_rate, loss,
val_bpm_output_loss, val_bpm_output_mae,
val_key_output_accuracy, val_key_output_loss, val_loss
```

### v4 Sonuç Özeti (Güncel)
- **Epoch sayısı:** 75 (early stopping, patience=20)
- **En iyi val_key_output_accuracy:** 0.477 (24 sınıf)
- **En iyi val_bpm_output_mae:** ~0.0095 (× 200 = ~1.9 BPM)
- **Parametre sayısı:** 858.009
- **Rastgele baseline:** %4.2 → model **11× daha iyi**

### Versiyon Karşılaştırması

| Versiyon | Mimari | Val Key Acc | Val BPM MAE | Not |
|----------|--------|-------------|-------------|-----|
| v1 | 3-blok CNN | %35.3 | ~2.7 BPM | İlk çalışan model |
| v2 | Residual + SE | ~%40 | ~2.2 BPM | HPSS eklendi |
| v3 | Dual-branch | — | — | Başarısız, overfitting |
| **v4** | **Hibrit GAP+GMP + TTA** | **%47.7** | **~1.9 BPM** | **Aktif model** |

> **Önemli Not (Keras 3.x class_weight fix):**  
> Keras 3.x `class_weight` parametresini multi-output modellerde desteklemez.
> Fix: `class_weight` → `sample_weight` olarak `tf.data.Dataset`'in 3. elemanına gömüldü:
> ```python
> # Eski (hata veriyor — Keras 3.x'te çalışmaz):
> model.fit(..., class_weight={"key_output": weights})
>
> # Yeni (çalışıyor):
> dataset = tf.data.Dataset.from_tensor_slices((X, y, sample_weights))
> model.fit(dataset)  # class_weight parametresi yok
> ```

---

## 10. Yaygın Görevler (Common Tasks)

### Modeli yeniden eğit (feature extraction değişmeden)
```bash
python src/train.py --epochs 120 --batch 16
```

### Feature extraction değişince cache yenile
```bash
python src/train.py --recache --epochs 120
```

### Tek dosya test
```bash
python src/predict.py data/raw/previews/herhangi.mp3
```

### Model mimarisini gör
```bash
python src/model.py
```

### Eğitim ilerlemesini takip et
```bash
tail -f training_output.log
# veya
python src/monitor.py   # web dashboard: http://localhost:8765
```

### Word raporu üret
```bash
python generate_report.py
# Çıktı: TrackBang_Key_Detection_Rapor.docx
```

### requirements.txt kur
```bash
python3.9 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## 11. Bilinen Kısıtlamalar ve Gelecek Çalışmalar

### Mevcut Kısıtlamalar
- 24-sınıflı problem: Bazı sınıflarda yetersiz örnek (min: C# Minör, 83 örnek)
- 30 saniyelik önizleme: Tam parça yerine kısaltılmış ses
- Elektronik müzik odaklı: Klasik veya akustik müzikte performans düşebilir
- Tek feature branch: Yalnızca chroma, mel-spectrogram entegre değil

### Planlanan İyileştirmeler
1. Mel-Spectrogram + Chroma dual-branch modeli
2. Dataset genişletme (10.000+ şarkı hedefi)
3. Camelot wheel'e duyarlı custom loss fonksiyonu
4. TFLite dönüşümü (mobil inference)
5. Streaming API (gerçek zamanlı analiz)
6. TrackBang backend tam entegrasyonu

---

## 12. Bağımlılıklar

| Kütüphane | Versiyon | Kullanım |
|-----------|---------|----------|
| tensorflow | ≥2.12 | Model eğitimi ve inference |
| librosa | ≥0.10 | Audio yükleme, HPSS, Chroma CENS |
| numpy | ≥1.23 | Array işlemleri |
| scikit-learn | ≥1.2 | train_test_split, stratified sampling |
| fastapi | ≥0.100 | REST API sunucusu |
| uvicorn | ≥0.22 | ASGI server |
| pydantic | ≥2.0 | Response modelleri |
| tqdm | ≥4.65 | Progress bar |
| pymongo | ≥4.3 | MongoDB bağlantısı (build_dataset) |
| yt-dlp | ≥2023.7 | SoundCloud MP3 indirme |

Python versiyonu: **3.9**

---

## 13. Proje Bağlamı: TrackBang CUE Entegrasyonu

Bu model, TrackBang'in CUE AI DJ Asistanı ile şu şekilde entegre olur:

**CUE sistemi (`/var/www/trackbang-backend/routes/cueRoutes.js`):**
- Groq LLM ile kullanıcı mesajını parse eder
- 10 farklı intent: music, explain, navigate, top50, playlist_lookup, feedback...
- Camelot uyumluluk algoritması: komşu key'ler, paralel majör/minör
- Energy curve sıralama: rising, falling, peak, wave (BPM bazlı)
- Redis: 24 saatlik dedup (son 200 parça)

**Bu modelin katkısı:**
- MongoDB'deki şarkıların BPM ve Camelot etiketleri bu model tarafından üretilecek
- Şu an: 3.279 parça manuel olarak DJ'ler tarafından etiketlenmiş
- Hedef: Yeni yüklenen her şarkı otomatik olarak analiz edilsin

---

*Son güncelleme: Haziran 2026 | Yusuf Kerim Sarıtaş*
