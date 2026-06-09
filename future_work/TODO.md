# Gelecek Çalışmalar ve Yapılacaklar

Bu belge, projeyi devralacak geliştirici veya araştırmacıya yol göstermek için hazırlanmıştır.

---

## Öncelikli — Kısa Vadeli

### Model Performansı
- [ ] **Camelot-farkında kayıp fonksiyonu:** Komşu tonlar (8A → 9A) arasındaki tahmin hatası, uzak tonlara (8A → 2B) göre daha az cezalandırılmalı. Mevcut CrossEntropy tüm hataları eşit tutar.
- [ ] **Dataset genişletme:** 3.817 → 10.000+ şarkı. Özellikle az örnekli sınıflar (C# Minor: 83, D# Minor: 97) için veri toplanmalı.
- [ ] **Konfidans kalibrasyon:** Softmax olasılıkları her zaman gerçek güveni yansıtmaz. Temperature scaling denenebilir.

### Mühendislik
- [ ] **TrackBang backend entegrasyonu:** Yeni şarkı yüklendiğinde otomatik analiz tetiklenmeli (`POST /predict` → MongoDB güncelle).
- [ ] **Model versiyonlama:** `models/` altında hangi modelin aktif olduğunu belirten `model_config.json` dosyası.
- [ ] **Birim testleri:** `src/` altındaki her fonksiyon için pytest testleri.

---

## Orta Vadeli

### Mimari İyileştirmeler
- [ ] **Transformer encoder:** CNN özelliklerinin üzerine multi-head self-attention bloğu. Zaman boyutundaki uzun menzilli bağımlılıkları yakalayabilir.
- [ ] **Mel spektrogram + Chroma fusion:** v3'te başarısız olan dual-branch yaklaşım, daha dikkatli kayıp dengeleme ile yeniden denenmeli.
- [ ] **Contrastive learning:** Aynı tondaki parçaları embedding uzayında yaklaştıran ön-eğitim stratejisi.

### Veri
- [ ] **Veri kalite filtresi:** SoundCloud kaynaklı etiketler için güven skoru. Düşük güvenilirli örnekleri eğitimden çıkar veya ağırlığını azalt.
- [ ] **Cross-genre değerlendirme:** Klasik, jazz, akustik müzikte performans ölçümü (şu an yalnızca elektronik).

### API
- [ ] **Gerçek zamanlı analiz:** WebSocket ile canlı ses akışından key/BPM tespiti.
- [ ] **Toplu işlem endpoint'i:** `POST /predict/batch` — birden fazla MP3 aynı anda.

---

## Uzun Vadeli — Araştırma

### Yeni Özellik Uzayları
- [ ] **Tonnetz özelliği:** Harmonic pitch-class uzayı, key değişimlerini daha iyi temsil edebilir.
- [ ] **Self-supervised ön-eğitim:** Etiketsiz büyük müzik veri setiyle (Million Song Dataset) ön-eğitim, ardından fine-tuning.

### Üretime Taşıma
- [ ] **TFLite dönüşümü:** Mobil cihazda offline tahmin için model optimizasyonu ve niceleme (quantization).
- [ ] **ONNX export:** Framework bağımsız deployment, diğer backend'lerde kullanım.
- [ ] **Model izleme:** Production'da tahmin dağılımını izle, drift tespiti.

### Camelot Wheel Zekası
- [ ] **Akustik Camelot benzerliği:** İki parça arasındaki harmonik geçiş kalitesini sayısal olarak ölç.
- [ ] **Set planlama optimizasyonu:** Verilen N parça arasından en uyumlu sıralamayı bulan algoritma.

---

## Bilinen Hatalar / Teknik Borç

- [ ] `build_dataset.py` SSH tünel adresini hard-code ediyor — config dosyasına taşınmalı
- [ ] `scraper.py` hata yönetimi yetersiz — ağ hatalarında sessizce devam ediyor
- [ ] `results/training_log.csv` her eğitimde üzerine yazılıyor — tarihli yedek alınmalı
- [ ] `venv/` klasörü `.gitignore`'da ama büyük model dosyaları Git LFS'e taşınmamış

---

## Katkı Yapmak İsteyenler İçin

1. Yukarıdaki maddelerden birini seç
2. GitHub Issue aç: hangi maddeyi ele aldığını belirt
3. `feature/konu-aciklamasi` branch'i oluştur
4. Değişikliği yap, CHANGELOG.md'ye ekle
5. Pull Request aç

Her katkı, gerçek bir üretim sistemine (TrackBang) doğrudan etki eder.
