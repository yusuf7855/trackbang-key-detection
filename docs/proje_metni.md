# Proje Metni

**Proje Adı:** TrackBang Audio Intelligence — Müzikal Ton ve BPM Tespiti  
**Öğrenci:** Yusuf Kerim Sarıtaş (221118047)  
**Danışman:** Dr. Öğr. Üyesi Nurettin Şenyer  
**Bölüm:** Yazılım Mühendisliği, Samsun Üniversitesi  
**Dönem:** 2025–2026 Bahar

---

## Özet

Bu proje, elektronik müzik parçalarının 30 saniyelik ses önizlemesinden **müzikal ton (key)** ve **tempo (BPM)** bilgisini otomatik olarak tahmin eden derin öğrenme tabanlı bir sistem geliştirmeyi amaçlamaktadır.

Sistem, iOS ve Android'de yayında olan **TrackBang** DJ müzik platformuna entegre edilmek üzere tasarlanmıştır. TrackBang'in yapay zeka destekli DJ asistanı **CUE**, kullanıcıların doğal dil ile harmonik set talep etmesine olanak tanır. Bu modelin ürettiği key/BPM etiketleri, CUE'nun Camelot uyumluluk algoritmasının temel girdisini oluşturmaktadır.

---

## Teknik Yaklaşım

### Özellik Çıkarma
- **HPSS** (Harmonic-Percussive Source Separation) ile perküsif gürültü temizleme
- **Chroma CENS** ile 12×1292 boyutlu özellik matrisi elde etme
- Z-score normalizasyon ve `.npy` önbellek sistemi

### Model Mimarisi (v4 — Hibrit CNN)
- Residual bloklar + SE (Squeeze-and-Excitation) Attention
- Paralel GAP + GMP (Global Average + Max Pooling) dalları
- Çoklu görev çıktısı: 24 sınıflı key sınıflandırması + BPM regresyonu
- 858.009 eğitilebilir parametre

### Eğitim
- **Veri seti:** 3.817 etiketli elektronik müzik parçası
- **Augmentasyon:** ±3 yarım ton pitch-shift, zaman/frekans maskeleme, Gaussian gürültü
- **Optimizasyon:** Cosine Decay öğrenme hızı, sınıf ağırlıklandırma
- **Test-Time Augmentation (TTA):** 5 tahmin ortalaması

### Sonuçlar (v4)
| Metrik | Değer |
|--------|-------|
| Validation Key Accuracy | %47.7 |
| Validation BPM MAE | ~1.9 BPM |
| Epoch sayısı | 75 (early stopped) |
| Rastgele tahmin (baseline) | %4.2 |

---

## Sistem Entegrasyonu

```
[Mobil App — Flutter iOS/Android]
         │ HTTP
         ▼
[Backend — Node.js + Express + MongoDB]
         │
         ├── CUE AI (Groq LLM + ruleParse.js)
         │        └── Camelot uyumluluk algoritması
         │
         └── Key/BPM API (bu proje — FastAPI)
                  └── key_detection_v4_best.keras
```

REST API: `POST /predict` — MP3 yükle, key + BPM + Camelot kodu al

---

## Kullanılan Teknolojiler

| Kategori | Teknoloji |
|----------|-----------|
| Derin Öğrenme | TensorFlow 2.12+, Keras |
| Ses İşleme | Librosa, HPSS |
| API | FastAPI, Uvicorn |
| Veri | MongoDB, SoundCloud (yt-dlp) |
| Dil | Python 3.9 |

---

## Özgün Katkılar

1. Elektronik müziğe özel eğitilmiş ilk açık kaynak key/BPM modeli
2. Camelot wheel uyumlu 24 sınıflı tahmin sistemi
3. HPSS + Chroma CENS kombinasyonunun DJ müziğine uygulanması
4. Gerçek üretim ortamına entegre edilmiş çalışan sistem (TrackBang API)
5. CUE AI asistanı ile uçtan uca harmonik set oluşturma pipeline'ı
