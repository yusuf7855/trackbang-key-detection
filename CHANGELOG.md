# Değişiklik Günlüğü (CHANGELOG)

Bu proje [Semantic Versioning](https://semver.org/) standardını takip eder.

---

## [v4.0.0] — Haziran 2026 — Hibrit CNN + Çoklu Görev Öğrenimi

### Eklendi
- Hibrit CNN mimarisi: GAP (Global Average Pooling) + GMP (Global Max Pooling) paralel dalları
- Test-Time Augmentation (TTA): her tahmin için 5 augmented kopya ortalaması
- ±3 yarım ton pitch-shift augmentasyonu (eğitim setinde on-the-fly)
- Çoklu görev öğrenimi iyileştirmesi: key ve BPM kayıpları ayrı ağırlıklarla dengelendi
- `key_detection_v4_best.keras` model dosyası

### Değişti
- Model parametreleri: ~2.97M → ~858K (daha verimli)
- Validation accuracy: %35.3 (v1) → %47.7 (v4, epoch 75/100)
- BPM MAE: ~2.7 → ~1.9 BPM

### Sonuçlar
- **val_key_output_accuracy:** %47.7
- **val_bpm_output_mae:** ~1.9 BPM
- **Eğitim:** 75 epoch (early stopping, patience=20)

---

## [v3.0.0] — Mayıs 2026 — Mel Spektrogram Denemesi

### Eklendi
- Mel spektrogram özellik dalı (Chroma CENS ile paralel)
- Dual-branch özellik çıkarma deneyi

### Sonuçlar
- Chroma CENS tek başına daha iyi sonuç verdi
- Dual-branch yaklaşım overfitting'e yol açtı
- Deneme başarısız, v4'te tek dal korundu

---

## [v2.0.0] — Nisan 2026 — Residual + SE Attention

### Eklendi
- HPSS (Harmonic-Percussive Source Separation) ön işleme adımı
- Residual bloklar (skip connections)
- SE (Squeeze-and-Excitation) attention mekanizması
- Cosine Decay öğrenme hızı takvimi
- Frekans maskeleme augmentasyonu
- `sample_weight` ile sınıf dengeleme (Keras 3.x uyumlu)

### Değişti
- Filtre sayısı: 32→64→128 → 64→128→256→256
- Early stopping patience: 15 → 20
- Mimari derinliği: 3 blok → 4 blok

### Sonuçlar
- **val_key_output_accuracy:** ~%40 civarı
- Keras 3.x `class_weight` hatası tespit edildi ve `sample_weight` ile düzeltildi

---

## [v1.0.0] — Ocak 2026 — İlk Model

### Eklendi
- Temel 3-blok CNN mimarisi
- Chroma CENS özellik çıkarma
- Çoklu görev çıktısı: key (24 sınıf) + BPM (regresyon)
- `.npy` önbellek sistemi
- FastAPI ile REST endpoint (`/predict`, `/health`)
- `labels.csv` — 3.817 etiketli şarkı

### Sonuçlar
- **val_key_output_accuracy:** %35.3 (epoch 39, early stopped)
- **val_bpm_output_mae:** ~2.7 BPM
- Rastgele tahminin (%4.2) çok üzerinde, ancak iyileştirme gerekiyor

---

## [v0.1.0] — Aralık 2025 — Veri Toplama

### Eklendi
- MongoDB → SoundCloud pipeline (`build_dataset.py`)
- `yt-dlp` ile 30 saniyelik MP3 önizleme indirme
- `scraper.py` ile ek veri toplama
- İlk `labels.csv`: 2.191 MP3, 3.817 etiket
