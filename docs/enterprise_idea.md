# Kurumsal Mekan AI Müzik Platformu
### Karpathy Tarzı Araştırma Fikir Belgesi

> **Yazar:** Yusuf Kerim Sarıtaş  
> **Danışman:** Dr. Öğr. Üyesi Nurettin Şenyer  
> **Tarih:** Haziran 2026  
> **Durum:** Erken aşama araştırma hipotezi

---

## 0. Tek Cümlelik Özet

> Herhangi bir ticari mekan için mekan türü, müşteri demografisi, günün saati ve marka kimliğine göre gerçek zamanlı kişiselleştirilen; lisans maliyetlerini sıfıra indiren ve DJ kalitesinde ses atmosferi sunan, telif hakkı olmayan harmonik açıdan tutarlı arka plan müzik setleri üreten bir yapay zeka sistemi.

---

## 1. Problem Tanımı

### 1.1 Temel Problem

Kafeler, restoranlar, perakende mağazalar, oteller, hastaneler — tüm ticari mekanlar arka plan müziğine ihtiyaç duyar. Ancak bu mekanlar acımasız bir ikilemle karşı karşıyadır:

**Seçenek A:** Spotify/YouTube kullan → Türkiye'de ticari kullanım için yasadışı (5846 Sayılı Kanun), MESAM/MÜ-YAP cezası riski  
**Seçenek B:** Lisanslı arka plan müziği servisi için ödeme yap → pahalı, jenerik, özelleştirme yok  
**Seçenek C:** Sessiz kal → korkunç müşteri deneyimi, mekanda geçirilen süreyi ve satışları ölçülebilir şekilde düşürür

Bu seçeneklerin hiçbiri iyi değil. Henüz mevcut olmayan daha iyi bir dördüncü seçenek var.

### 1.2 Gerçekte Eksik Olan Nedir?

Mevcut arka plan müziği servisleri *lisanslama* sorununu çözüyor ama *kaliteyi* görmezden geliyor. Yasal açıdan güvenli, ruh haline uygun müzik çalıyorlar — ancak:

- Parçalar arası geçişler ani ve rahatsız edici (uyumsuz ton değişimleri, BPM sıçramaları)
- Gün boyunca enerji eğrisine uyum yok (sabah açılış → öğlen yoğunluk → akşam kapanış)
- Harmonik tutarlılık yok — dizi düzgün bir set gibi değil, karışık bir playlist gibi duyuluyor
- Demografik uyum yok — eczanedeki 50+ yaşındaki müşteri, sokak giyimi mağazasındaki 20 yaşındakiyle aynı playlist'i dinliyor
- Müzik çoğunlukla AI üretimi ve "yapay" bir his veriyor; bu durum farkında olmadan marka kalitesini düşürüyor

Boşluk: **Hiç kimse ticari mekan müziğine DJ kalitesinde harmonik kürasyon uygulamıyor.**

### 1.3 Neden Şimdi?

2025–2026'da üç güç bir araya geliyor:

1. **Müzik için üretken AI olgunlaştı.** Meta MusicGen, Suno v3, Udio ve açık kaynak alternatifleri artık büyük ölçekte üretilmiş parçalardan ayırt edilemeyen tür/ruh hali/tempo odaklı ses üretebiliyor.
2. **Telif hakkı yaptırımları sıkılaştı.** MESAM ve MÜ-YAP, Türkiye'deki ticari mekanlara yönelik denetimleri önemli ölçüde artırdı. Gizli korku artık somut bir tehdit haline geldi.
3. **TrackBang'in mevcut altyapısı.** Çalışan bir key/BPM tespit modeli (%47.7 doğruluk, 24 sınıf), Camelot uyumluluk motoru ve üretim backend'i zaten mevcut. Bunu inşa etmenin marjinal maliyeti sıfırdan başlayan bir girişime kıyasla çok daha düşük.

---

## 2. Önerilen Çözüm

### 2.1 Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────┐
│                    MEKAN PROFİLİ                             │
│  mekan_turu | yas_araligi | tema | saat | doluluk | sezon   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│           SET PLANLAMA MOTORU (LLM + Kurallar)               │
│                                                             │
│  Girdi:  mekan profili + hedef enerji eğrisi                 │
│  Çıktı:  [ {ton, bpm, ruh_hali, süre}, ... ] × N parça      │
│                                                             │
│  Kısıtlar:                                                  │
│    - Camelot wheel: yalnızca komşu tonlar (±1 adım)         │
│    - BPM farkı: ardışık parçalar arası maks ±8 BPM          │
│    - Enerji eğrisi: açılış→kapanış saatlerine kosinüs eğri  │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────┴──────────────┐
            │                            │
            ▼                            ▼
┌───────────────────┐        ┌───────────────────────┐
│   MÜZİK KÜTÜPHANESİ│       │   AI MÜZİK ÜRETECİ    │
│   (sahip olunan)  │        │   (MusicGen / Suno)   │
│                   │        │                       │
│  - Tür etiketli   │        │  Prompt: "ambient     │
│  - Ton/BPM bilgili│        │  lounge jazz, 92 BPM, │
│  - Ruh hali skoru │        │  A minör, 3 dk"       │
│  - Haklar bizde   │        │                       │
└────────┬──────────┘        └──────────┬────────────┘
         │                             │
         └──────────┬──────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│              SES SON İŞLEME                                │
│                                                           │
│  - Parçalar arası EQ normalizasyonu                       │
│  - Crossfade geçiş (harmonik mix, 4–8 bar örtüşme)        │
│  - Ses seviyesi normalizasyonu (EBU R128 / -14 LUFS)      │
│  - Opsiyonel: yapılandırılabilir slotlara marka jingle    │
└───────────────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│              DAĞITIM                                       │
│                                                           │
│  - Web Player (tarayıcı tabanlı, sıfır donanım)           │
│  - Smart Stream (uyarlanabilir bit hızı)                  │
│  - Çevrimdışı Radyo Player (internet kesilse de devam)    │
│  - REST API (POS sistemi entegrasyonu, IoT sensörler)     │
└───────────────────────────────────────────────────────────┘
```

### 2.2 Camelot Harmonik Motoru (Temel Farklılaştırıcı)

Camelot Wheel, 24 müzikal tonu saat kadranı biçiminde bir ızgaraya eşler. Komşu tonlar (±1 konum) harmonik olarak uyumludur — aralarındaki geçişler insan kulağına akıcı gelir.

```
        12B  1B  2B
    11B           3B
  10B               4B
   9B               5B
    8B           6B
        7B  6B  5B   ← iç halka = minör tonlar (A)
```

**Mevcut durum (TrackBang v4):** Ton tespit modeli herhangi bir 30 saniyelik ses klibini %47.7 top-1 doğrulukla sınıflandırıyor. Bu geçiş planlaması için yeterli — hatalar olsa bile Camelot komşuluk kısıtlamaları harmonik uyumsuz geçişleri engelliyor.

**Mekan kullanımı için:** Yüklenen parçaların tonlarını tespit etmek yerine sistem istenen tonda parçalar *üretiyor*. Bu, temel harmonik pipeline için tespit hatasını tamamen ortadan kaldırıyor.

### 2.3 Mekan Profili Şeması

```json
{
  "mekan": {
    "tur": "kafe | restoran | perakende | otel_lobi | spa | spor_salonu | hastane | eczane | ofis",
    "marka_seviyesi": "butce | orta | premium | luks",
    "birincil_yas_araligi": "18-25 | 25-35 | 35-50 | 50+",
    "tema": "string (örn. 'endüstriyel minimalist', 'akdeniz', 'sokak giyimi')"
  },
  "program": {
    "acilis": "09:00",
    "kapanis": "22:00",
    "yogun_saatler": ["12:00-14:00", "18:00-20:00"],
    "enerji_egrisi": "sabah_sakin | sabit | yukselis | aksam_pik"
  },
  "muzik_tercihleri": {
    "turler": ["lounge", "deep_house", "akustik_pop"],
    "haric_turler": ["metal", "kufurlu_rap"],
    "bpm_araligi": [80, 120],
    "dil_tercihi": "enstrumantal | turkce | ingilizce | karma"
  },
  "marka": {
    "jingle_aktif": true,
    "anons_slotlari": ["12:00", "18:00"],
    "ozel_ses_id": "string"
  }
}
```

### 2.4 Enerji Eğrisi Haritalaması

```
Enerji
  1.0 │                              ╭──╮
  0.9 │                         ╭───╯  ╰──╮
  0.8 │                    ╭───╯          ╰──
  0.7 │               ╭───╯
  0.6 │          ╭───╯
  0.5 │     ╭───╯
  0.4 │╭───╯
  0.3 ╰┤
      └──────────────────────────────────────── Saat
      09:00  11:00  13:00  15:00  17:00  19:00  21:00

Enerji = f(BPM, ton_modu, üretim_yoğunluğu)
  BPM:      80 BPM (açılış) → 128 BPM (pik) → 95 BPM (kapanış)
  Mod:      Majör (enerjik) ↔ Minör (samimi/karanlık)
  Yoğunluk: seyrek akustik → tam elektronik
```

---

## 3. Pazar Analizi

### 3.1 Global Pazar Büyüklüğü

| Segment | 2025 Değeri | 2031 Tahmini | CAGR |
|---------|------------|-------------|------|
| Arka Plan Müziği (Global) | 3,0 Milyar $ | 5,1 Milyar $ | %9,6 |
| Ticari Arka Plan Müziği | 2,04 Milyar $ | 2,78 Milyar $ | %6,35 |
| Müzikte AI (Genel) | 4,48 Milyar $ | 8,5 Milyar $ | %23,7 |

Kaynak: Polaris Market Research, Mordor Intelligence, Sound Verse AI (2025)

### 3.2 Türkiye Pazar Büyüklüğü (Aşağıdan Yukarıya Tahmin)

```
Türkiye'deki Kafe ve Restoranlar:      ~350.000
Perakende Mağazalar (zincir):          ~85.000
Oteller (3-5 yıldız):                  ~4.500
Hastane ve Klinikler:                  ~35.000
Kurumsal Ofisler (50+ kişi):           ~120.000
──────────────────────────────────────────────
Toplam Ulaşılabilir Mekan:             ~594.500

Gerçekçi penetrasyon (3. Yıl):         %1,5 → ~8.900 mekan
Ortalama Aylık Gelir (ARPU):           ₺500/ay
Yıllık Gelir Potansiyeli (3. Yıl):     ₺53M/yıl (~1,6M USD)
```

**Not:** Bunlar muhafazakâr tahminler. SMG şu an 13.000+ Türk mekana hizmet veriyor. Bu ölçeğin %50'sini yakalamak yılda ₺78M demek.

### 3.3 Müşteri Segmentasyonu

**Segment 1 — KOBİ Mekanlar (birincil hedef)**
- Bağımsız kafeler, restoranlar, berberler
- Acı nokta: Spotify yasal değil ama yine de kullanıyorlar; MESAM cezası korkusu
- Ödeme isteği: ₺300–600/ay
- Karar verici: işletme sahibi (tek kişi)
- Mesaj: "Yasal, uygun fiyatlı, Spotify'dan daha iyi ses"

**Segment 2 — Perakende Zincirleri (ölçek hamlesi)**
- LCW, DeFacto, Zara TR, Boyner, MediaMarkt TR
- Acı nokta: yüzlerce mağazada merkezi kontrol, marka tutarlılığı
- Ödeme isteği: ₺200–400/mağaza/ay (×yüzlerce mağaza)
- Karar verici: pazarlama direktörü
- Mesaj: "Tek dashboard, 500 mağaza, markaya özel ses"

**Segment 3 — Premium Konaklama (premium konumlanma)**
- 5 yıldızlı oteller, gurme restoranlar, lüks spalar
- Acı nokta: mevcut çözümler marka seviyelerine göre jenerik kalıyor
- Ödeme isteği: ₺2.000–8.000/ay
- Karar verici: F&B direktörü veya marka yöneticisi
- Mesaj: "Kendi radyo istasyonunuz. Markanız gibi ses çıkarıyor."

**Segment 4 — Sağlık Sektörü (uzun vadeli, hizmet almayan)**
- Hastaneler, klinikler, eczaneler, bekleme odaları
- Acı nokta: jenerik "hastane müziği" algılanan bakım kalitesini düşürüyor
- Araştırma desteği: 2023 NIH çalışması müziğin hasta kaygısını %37 azalttığını doğruluyor
- Ödeme isteği: ₺800–2.000/ay
- Karar verici: hastane yöneticisi
- Mesaj: "Klinik temelli müzik hasta stresini azaltır"

---

## 4. Rekabet Analizi

### 4.1 Global Rakipler

| Şirket | Kuruluş | Ölçek | AI? | Harmonik Kürasyon? | Zayıflık |
|--------|---------|-------|-----|-------------------|---------|
| **Mood Media** (eski Muzak) | 1934 | 500K+ mekan, 140 ülke | Sınırlı | Hayır | Eski teknoloji, pahalı, Türkiye ofisi yok |
| **Soundtrack Your Brand** | 2013 | Türkiye'de aktif | Playlist AI (2024) | Hayır | Spotify'a bağımlı, lisans maliyeti müşteriye yansıyor |
| **Brandtrack** | 2015 | McDonald's, Hilton | AI + insan hibrit | Hayır | Premium fiyat, KOBİ'ye uygun değil |
| **Rockbot** | 2011 | 1M+ kullanıcı | Öneri AI | Hayır | ABD odaklı, Türkçe destek yok |
| **PlayNetwork** | 1996 | Starbucks ortağı | Hayır | Hayır | Yalnızca büyük kurumlar için |

**Temel global tespit:** Hiçbir büyük oyuncu arka plan müziği programlamasında harmonik uyumluluk (Camelot wheel) uygulamıyor. Hepsi parçaları harmonik bir dizi olarak değil, bağımsız birimler olarak ele alıyor.

### 4.2 Türk Rakipler

| Şirket | Ölçek | AI? | Harmonik Kürasyon? | Temel Zayıflık |
|--------|-------|-----|-------------------|--------------|
| **SMG** | 13.000+ TR mekan | Temel programlama | Hayır | Müzik dış kaynaklı, üretilmiyor; jenerik |
| **Mağaza Radyo** | 5.700 mekan | Hayır | Hayır | Saf playlist servisi, teknoloji farklılaşması yok |
| **Kurumsal Müzik Net** | Bilinmiyor | AI üretim ✓ | Hayır | Müzik üretiyor ama kürasyon zekası yok |
| **VIVA MEDYA** | Küçük | Hayır | Hayır | Kurumsal radyo odaklı, mekan özelinde değil |
| **NOT FM / MAGAZA FM** | Küçük | Hayır | Hayır | Yalnızca tür bazlı |

**Temel Türk tespiti:** Kurumsal Müzik Net en yakın rakip — AI üretimi ile lisanslama sorununu çözüyorlar. Ama harmonik kürasyon katmanları yok. Ürünleri "yasal müzik", "zekice sıralanmış yasal müzik" değil.

### 4.3 Rekabet Konumlanma Haritası

```
                    YÜKSEK HARMONİK KALİTE
                           │
                    HEDEF  │◀── Bizim konumumuz
                           │
    JENERİK ───────────────┼─────────────────── KİŞİSELLEŞTİRİLMİŞ
    MÜZİK                  │                         MÜZİK
                           │
    Kurumsal  SMG  Mağaza  │        Soundtrack    Brandtrack
    Müzik Net      Radyo   │        Your Brand
                           │
                    DÜŞÜK HARMONİK KALİTE
```

### 4.4 Sürdürülebilir Rekabet Avantajları

1. **Tescilli harmonik kürasyon motoru** — herhangi bir Türk rakibin inşa etmesi 2+ yıl alır
2. **Sahip olunan müzik kütüphanesi** — her üretimle büyür; rakipler yıllarca üretim yapmadan bunu kopyalayamaz
3. **Ton tespit altyapısı** — zaten inşa edildi (TrackBang v4 modeli); rakiplerin sıfırdan inşa etmesi gerekir
4. **Veri volan etkisi** — daha fazla mekan → daha fazla davranış verisi → daha iyi kişiselleştirme → daha fazla mekan
5. **Türk pazarı ilişkisi** — yerel dil, yerel destek, Türk müzik kültürü anlayışı

---

## 5. SWOT Analizi

### Güçlü Yönler
- **Çalışan key/BPM tespit modeli** (858K parametre, %47.7 doğruluk, üretimde konuşlandırıldı)
- **Camelot harmonik algoritması** zaten üretimde (TrackBang CUE AI)
- **FastAPI altyapısı** — key tespit API'si çalışıyor; genişleme aşamalı
- **Kurucu DJ** — 7+ yıl alan uzmanlığı; çıktı kalitesini kulağıyla değerlendirebilir
- **Sıfır harici lisanslama** — AI üretimli müzik modeli MESAM/MÜ-YAP bağımlılığını ortadan kaldırır
- **İlk hareket avantajı** — Türkiye'de harmonik farkındalığa sahip mekan müziğinde

### Zayıf Yönler
- **AI üretimli müzik kalite açığı** — mevcut üretken modeller (MusicGen, Suno) iyi ama mükemmel değil; bazı mekanlar gerçek sanatçılara kıyasla düşük marka kalitesi algılayabilir
- **Soğuk başlangıç problemi** — ürün satılabilir olmadan önce büyük ve çeşitli bir müzik kütüphanesi oluşturmak gerekiyor
- **Satış zorlu** — Türk KOBİ'lere B2B SaaS satışı saha satışı gerektiriyor; salt ürün odaklı büyüme mümkün değil
- **B2B alanında marka bilinirliği yok** — TrackBang DJ topluluğunda biliniyor, perakende/konaklama sektöründe değil
- **Donanım/çevrimdışı gereksinimi** — mekanlar çevrimdışı oynatma istiyor; gömülü yazılım veya donanım cihazı gerekiyor

### Fırsatlar
- **MESAM yaptırım dalgası** — artan cezalar, Spotify'ı yasadışı kullanan mekanların %90'ı için akut acı yaratıyor
- **Sağlık sektörü** — tamamen hizmet almayan; hiçbir rakibin burada ürünü yok
- **Uluslararası genişleme** — MENA/Türk dili konuşan pazarlarda Türkçe avantajı (Azerbaycan, Kazakistan, Özbekistan)
- **API/white-label hamlesi** — harmonik motoru teknoloji eksikliği olan ama dağıtım ağı geniş servislere (Mağaza Radyo, SMG) lisanslama
- **Üretken AI hızla gelişiyor** — müzik kalitesi 2–3 yıl içinde üretilmiş parçalarla eşitlik sağlayacak

### Tehditler
- **Mood Media'nın Türkiye'ye ciddi girişi** — sermaye ve ilişkileri var; harmonik kürasyon geliştirirse avantaj azalır
- **Spotify'ın Türkiye'de ticari tier başlatması** — Spotify bazı pazarlarda zaten ticari lisanslama sunuyor; AI playlist kürasyon ile Türkiye'ye genişlerse KOBİ segmenti zorlaşır
- **AI müzik telif hukuku değişiklikleri** — AI üretimli müzik toplu lisanslamaya tabi hale gelirse (küresel olarak hâlâ belirsiz) temel maliyet avantajı ortadan kalkar
- **Suno/Udio'nun doğrudan B2B sunması** — üretken AI müzik şirketleri kendi üretimleri üzerine mekan programlama inşa ederse ara katman atlanabilir

---

## 6. İş Modeli

### 6.1 Gelir Akışları

**Birincil: SaaS Aboneliği**
```
Paket        | Mekan | Fiyat/ay    | Özellikler
─────────────|────────|─────────────|──────────────────────────────────
Başlangıç    | 1      | ₺399        | Temel programlama, 3 mekan profili
Profesyonel  | 1-5    | ₺799        | + Anonslar, analitik, çevrimdışı
İş           | 5-50   | ₺299/mekan  | + API erişimi, özel ses, öncelik
Kurumsal     | 50+    | Özel fiyat  | + White-label, özel destek
```

**İkincil: Donanım Cihazı (opsiyonel)**
- Raspberry Pi tabanlı "TrackBang Box" — tak-çalıştır çevrimdışı oynatıcı
- Tek seferlik ₺1.500 donanım + ₺199/ay abonelik
- Teknik bilgisi olmayan mekan sahipleri için BT karmaşıklığını ortadan kaldırır

**Üçüncül: API Lisanslama**
- Harmonik programlama motorunu rakip servislere lisanslama
- ₺0,001/istek veya gelir paylaşımı modeli

### 6.2 Birim Ekonomisi (Muhafazakâr 2. Yıl Tahmini)

```
Müşteri Edinim Maliyeti (CAC):      ₺800  (saha satış + dijital)
Kullanıcı Başına Ortalama Gelir:    ₺550/ay
Brüt Marj:                          ~%78  (bulut + üretim maliyeti ~%22)
Geri Ödeme Süresi:                  ~1,5 ay
LTV (24 ay, %3/ay churn):           ₺10.200
LTV/CAC:                            12,75x  ← güçlü birim ekonomisi
```

### 6.3 Pazara Giriş Stratejisi

**Aşama 1 — Kıyıbaşı (1–6. Aylar)**
- Hedef: İstanbul'da 50 bağımsız kafe (Kadıköy, Beyoğlu, Nişantaşı)
- Kanal: Doğrudan erişim, DJ topluluğu referansları, TrackBang kullanıcı tabanı
- Teklif: 3 ay ücretsiz, ardından ₺399/ay
- Hedef: 6. aya kadar ürün-pazar uyum sinyali, 10 ödeme yapan müşteri

**Aşama 2 — Ölçekleme (7–18. Aylar)**
- Hedef: İstanbul + Ankara + İzmir, KOBİ segmenti
- Kanal: Dijital pazarlama ("MESAM ceza" hedefli Google Ads), Yemeksepeti/GetirYemek restoran ağlarıyla ortaklıklar
- Hedef: 500 ödeme yapan mekan, ₺275K MRR

**Aşama 3 — Kurumsal (18–36. Aylar)**
- Hedef: Perakende zincirleri (pazarlama direktörlerine yaklaşım)
- Kanal: LinkedIn erişimi, ticaret fuarları (RetailTurkey, HoReCa Istanbul)
- Hedef: 3 kurumsal sözleşme, 2.000+ toplam mekan

---

## 7. Teknik Araştırma Soruları

Bunlar çözülmesi gereken açık problemler. Cevaplarını henüz bilmiyorum.

**S1: Harmonik tutarlılık mekan KPI'larını gerçekte ne kadar iyileştiriyor?**
- Hipotez: Harmonik tutarlı arka plan müziği mekanda kalma süresini %8–15 artırıyor ve bilinçaltı rahatsızlığı azaltıyor
- Test yöntemi: Aynı mekan profiline sahip iki kafede A/B testi — biri Camelot sıralı müzik, diğeri rastgele playlist
- Ölçüm: POS verisi (ortalama çek tutarı), zaman damgalı giriş/çıkış sayımı (kamera + CV), personel öznel değerlendirmesi

**S2: Uygulanabilir bir ürün için minimum müzik kütüphanesi boyutu nedir?**
- Hipotez: Tür başına 500 parça (10 tür = 5.000 parça) 30 günlük tekrarsız kapsam için yeterli
- Kritik: Müşterinin aynı parçayı bir haftada iki kez duyması güveni yıkıyor
- Test yöntemi: Kütüphane boyutu kısıtlamaları göz önünde bulundurularak playlist üretiminin entropisi hesaplanır

**S3: MusicGen/Suno mekan için hazır ses üretebilir mi?**
- Bilinen sınırlılık: Mevcut açık kaynak üretken modeller çıktıların yaklaşık %3–5'inde artefakt (bozukluk, tempo kayması, doğal olmayan armoni değişimleri) üretiyor
- Test yöntemi: 100 mekan sahibiyle kör A/B testi: "Hangisi daha profesyonel hissettiriyor?"
- Kabul edilebilir eşik: Üretim kullanımı için <%1 artefakt oranı

**S4: Camelot motoru kusurlu ton merkezine sahip AI üretimli parçaları nasıl ele almalı?**
- AI üretimli müziğin zaman zaman belirsiz ton merkezi oluyor (özellikle ambient/atonal parçalar)
- Mevcut v4 ton tespit modeli (%47.7 doğruluk) elektronik müzik üzerinde eğitildi — AI üretimli ambient müziğe genellenebilir mi?
- Test yöntemi: MusicGen ile 500 parça üret, v4 modeliyle tespit et, manuel doğrula — bilinen ton istemlerinde hassasiyeti ölç

**S5: Sorunsuz crossfading için optimal geçiş mimarisi nedir?**
- Seçenek A: Sabit uzunluklu crossfade (4/4'te 8 bar = 16 saniye)
- Seçenek B: Beat senkronize crossfade (BPM hassas beat tespiti gerektirir)
- Seçenek C: AI üretimli geçiş segmenti (iki parça arasında 4 barlık köprü üret)
- Seçenek C en özgün ama büyük ölçekte kanıtlanmamış

**S6: Batı dışı gamları ve Türk klasik müzik taleplerini nasıl ele alacağız?**
- Camelot wheel Batı 12 ton eşit tamperaman sistemini varsayıyor
- Türk makam müziği mikrotonal gamlar kullanıyor — Camelot uygulanamaz
- Fırsat: Makam uyumlu bir sıralama sistemi inşa et (tamamen özgün, önceki çalışma yok)

---

## 8. Uygulama Yol Haritası

### MVP (3 ay) — Temel döngünün çalıştığını kanıtla

```
Hafta 1-2:   MusicGen entegrasyonu — prompt ile 100 test parçası üret
Hafta 3-4:   Üretilen parçalarda ton tespiti doğrulaması
Hafta 5-6:   Camelot sıralayıcı — mekan profili verilince sıralı parça listesi üret
Hafta 7-8:   Ses crossfading pipeline — sorunsuz geçişler
Hafta 9-10:  Web player — tarayıcı tabanlı, uygulama kurulumu gerekmiyor
Hafta 11-12: İstanbul'da 5 beta kafe pilot uygulaması
```

**MVP başarı kriterleri:**
- [ ] Mekan profilinden uçtan uca 1 saatlik set 2 dakikadan kısa sürede üretildi
- [ ] 10 dinleme kullanıcı testinde hiç rahatsız edici geçiş yok
- [ ] 5 kafe sahibi ücretsiz deneme sonrası ödeme yapmaya razı

### V1 Ürün (6 ay)
- Mekan dashboard'u (React)
- Programlama (haftalık program, enerji eğrisi editörü)
- Anons entegrasyonu (özel sesli TTS)
- Çevrimdışı oynatıcı (Electron uygulaması veya Raspberry Pi imajı)
- 10 tür × 500 parça kütüphanesi

### V2 Ürün (12 ay)
- Gerçek zamanlı doluluk uyumu (IoT sensör veya kamera entegrasyonu)
- Analitik (kalma süresi korelasyonu, müşteri geri bildirimi)
- Mekan sahipleri için mobil uygulama
- POS entegrasyonu için API
- Sağlık sektörü pilot uygulaması

---

## 9. Neden Zor Olacak

Zorluklar konusunda dürüst olmak istiyorum çünkü gerçekçilik olmadan iyimserlik işe yaramaz.

1. **Müzik kalitesi bir his meselesi, metrik değil.** "Müziği beğenmedim" diyen bir kafe sahibi, tüm KPI'lar iyi olsa bile ayrılır. Öznel kaliteyi A/B test etmek ve sistematik olarak iyileştirmek zor.

2. **Türkiye'de B2B KOBİ satışı ilişki odaklı.** Türk KOBİ sahipleri web sitelerinden değil, güvendikleri insanlardan satın alır. Bu, sahada bir satış ekibi gerektirir. Pahalı ve yavaş.

3. **Ürün ortam müziği, heyecan verici değil.** Hiç kimse bir kafe açıp arka plan müziği servisi hakkında tweet atmaz. Bu sessiz, görünmez bir araç — bu da kulaktan kulağa yayılımın yavaş ve PR'ın zor olduğu anlamına gelir.

4. **Üretken AI müziği hızla gelişiyor ama devler de.** Spotify, YouTube Music ve Apple Music hepsi AI'ya ağırlıklı yatırım yapıyor. Spotify Türkiye'de AI programlama ile ticari tier başlatırsa sonsuz müzik kütüphanesi ve global marka güveni var.

5. **Entegrasyon cehennemi.** Farklı mekanlarda farklı ses sistemleri, internet kalitesi ve teknik yetkinlik var. Türk kafesinde "sorunsuz çalışan" (tutarsız Wi-Fi ve teknik bilgisi olmayan sahibiyle) güvenilir çevrimdışı oynatma inşa etmek gerçek bir mühendislik zorluğu.

---

## 10. Önceki Çalışmalar ve Kaynaklar

### Akademik
- Tzanetakis, G. & Cook, P. (2002). Musical genre classification of audio signals. *IEEE Trans. Speech Audio Process.* — ses özelliği çıkarımı üzerine temel çalışma
- Müller, M. (2015). *Fundamentals of Music Processing.* Springer. — Chroma özellikleri, HPSS, ton tespiti
- Bitteur, H. ve ark. (2020). Computational analysis of harmonic compatibility in DJ mixing. — Camelot tarzı analiz için en yakın akademik emsal
- Li, T. ve ark. (2023). Context-aware music recommendation for commercial environments. *ISMIR 2023.* — mekan farkındalıklı öneri
- Schneider, S. ve ark. (2023). Music in healthcare settings: systematic review of clinical outcomes. *NIH PMC10380075.* — sağlık sektörü için kanıt tabanı

### Endüstri
- Soundtrack Your Brand (2024). AI Playlist Generator duyurusu. PR Newswire.
- Mood Media. (2025). Global arka plan müziği pazar varlığı. moodmedia.com
- SMG. (2025). Music for Business. smg.com.tr
- Kurumsal Müzik Net. (2025). AI-Destekli Müzik Üretimi. kurumsalmuzik.net
- Copet, J. ve ark. (2023). Simple and Controllable Music Generation (MusicGen). *NeurIPS 2023.* arXiv:2306.05284

### Pazar Verisi
- Polaris Market Research. (2025). Background Music Market Size Report 2025–2034.
- Mordor Intelligence. (2026). Commercial Background Music Market Report.
- Sound Verse AI. (2025). AI Music Industry Trends 2026.

---

## 11. Danışmana Açık Sorular

1. Ambient/arka plan müziği için **harmonik dizi optimizasyonu** üzerine önceden yayınlanmış akademik çalışma var mı (DJ miksajının aksine)? Yoksa bu yayımlanabilir bir katkı olabilir.

2. v4 ton tespit modeli AI üretimli müzik üzerinde ince ayar yapılmalı mı, yoksa dağılım farkı görmezden gelinebilecek kadar küçük mü?

3. **Türk makam uzantısı** (yukarıdaki S6) tez düzeyinde uygulanabilir bir katkı mı olur? Tamamen özgün olurdu.

4. A/B test metodolojisi için (S1), müzik psikolojisi literatüründe kalma süresi etkilerini ölçmek için standart bir protokol var mı?

---

---

## 12. Pilot Mekan Senaryosu: Tek Kafe — Uçtan Uca Somut Plan

> Bu bölüm, yukarıdaki mimari ve iş modelini **gerçek bir mekana** uygulandığında nasıl görüneceğini somutlaştırır. Varsayımsal ama gerçekçi bir örnek üzerinden yürütülmektedir.

### 12.1 Mekan Profili

| Alan | Değer |
|------|-------|
| **Mekan adı** | Frekans Kafe (varsayımsal) |
| **Konum** | Kadıköy, İstanbul |
| **Alan** | ~80 m², 30 oturma kapasitesi |
| **Açılış saatleri** | 08:00 – 23:00 (15 saat/gün) |
| **Müşteri profili** | 22–35 yaş, üniversite mezunu, freelancer/öğrenci ağırlıklı |
| **Mevcut müzik** | Spotify (yasadışı ticari kullanım) |
| **Mevcut ses sistemi** | 2× pasif hoparlör + mixer amplifikatör, 3.5mm/Bluetooth girişi var |
| **İnternet** | 50 Mbps fiber, ara sıra kesinti |
| **Teknik yetkinlik** | Düşük — sahibi yazılım bilmiyor |

**Mekan Profili JSON:**
```json
{
  "mekan": {
    "tur": "kafe",
    "marka_seviyesi": "orta",
    "birincil_yas_araligi": "22-35",
    "tema": "sade minimalist, çalışma dostu"
  },
  "program": {
    "acilis": "08:00",
    "kapanis": "23:00",
    "yogun_saatler": ["09:00-11:00", "14:00-17:00"],
    "enerji_egrisi": "sabah_sakin → oglen_orta → aksam_pik"
  },
  "muzik_tercihleri": {
    "turler": ["lo-fi hip hop", "deep house", "akustik pop", "chill electronic"],
    "haric_turler": ["metal", "arabesk", "yüksek enerjili EDM"],
    "bpm_araligi": [75, 118],
    "dil_tercihi": "enstrumantal"
  },
  "marka": {
    "jingle_aktif": false,
    "anons_slotlari": [],
    "ozel_ses_id": null
  }
}
```

---

### 12.2 Teknik Kurulum

#### Donanım (Tek Seferlik)

| Bileşen | Model / Açıklama | Maliyet |
|---------|-----------------|---------|
| Mini bilgisayar | Raspberry Pi 5 (8GB RAM) | ₺1.850 |
| Depolama | 128GB microSD (Class 10) | ₺180 |
| Ses çıkışı | USB DAC (3.5mm stereo çıkış) | ₺320 |
| Kasa + soğutucu | Aluminyum kasa + fan | ₺250 |
| Güç adaptörü | 27W USB-C | ₺120 |
| **Toplam donanım** | | **₺2.720** |

> **Alternatif:** Mekanda Wi-Fi üzerinden akış yeterli ise sadece tarayıcı tabanlı web player kullanılır — sıfır donanım maliyeti. Raspberry Pi yalnızca çevrimdışı güvenilirlik için gerekli.

#### Yazılım Yığını (Raspberry Pi üzerinde)

```
┌─────────────────────────────────────────────────┐
│              Raspberry Pi 5 OS (Lite)            │
│                                                 │
│  ┌──────────────────────────────────────────┐   │
│  │         TrackBang Player Servisi          │   │
│  │  - Python 3.11 + FastAPI (yerel sunucu)  │   │
│  │  - VLC (ses çıkışı motoru)               │   │
│  │  - SQLite (yerel playlist cache)         │   │
│  │  - 12 saatlik müzik buffer (SD kart)     │   │
│  └──────────────────────────────────────────┘   │
│                    │                            │
│         ┌──────────▼──────────┐                 │
│         │  Bulut Senkronizasyon│                 │
│         │  - Gece 02:00'da     │                 │
│         │    yeni set indir    │                 │
│         │  - TrackBang API →   │                 │
│         │    günlük playlist   │                 │
│         └─────────────────────┘                 │
└─────────────────────────────────────────────────┘
         │ 3.5mm stereo
         ▼
  [Mixer Amplifikatör] → [Hoparlörler]
```

#### Kurulum Adımları (Teknisyen 1 Ziyaret, ~2 Saat)

```
1. Raspberry Pi imajını yaz (önceden hazırlanmış, 10 dk)
2. Wi-Fi şifresi gir, cihazı internete bağla (5 dk)
3. TrackBang sunucusuna cihaz kaydı (QR kod tara, 2 dk)
4. Mekan profilini web panelden doldur (10 dk)
5. 3.5mm kabloyu mevcut amplifikatöre bağla (5 dk)
6. İlk playlist'i çek ve test et (10 dk)
7. Sahibine 3 şeyi öğret:
   - Ses seviyesi nasıl ayarlanır (fiziksel knob)
   - Müzik durdu mu? → cihazı yeniden başlat (1 buton)
   - Sorun olursa WhatsApp destek hattı
```

---

### 12.3 Günlük Müzik Programı

Mekan profiline göre sistem **3 farklı enerji bloğu** üretir:

| Saat Dilimi | BPM Aralığı | Ton Modu | Tür | Atmosfer |
|-------------|-------------|----------|-----|----------|
| 08:00 – 11:00 | 72–88 BPM | Majör | Lo-fi, akustik | Sakin, konsantre |
| 11:00 – 17:00 | 88–105 BPM | Karma | Chillout, deep house | Orta enerji |
| 17:00 – 23:00 | 100–118 BPM | Minör ağırlıklı | Melodic house, chill electronic | Sosyal, akşam |

**Harmonik geçiş örneği (akşam bloğu):**
```
Parça 1:  8A  — 104 BPM  (Am)
Parça 2:  9A  — 106 BPM  (Em)   ← Camelot +1 adım ✓
Parça 3:  9A  — 108 BPM  (Em)   ← Aynı ton, BPM artışı ✓
Parça 4:  10A — 108 BPM  (Bm)   ← Camelot +1 adım ✓
Parça 5:  10B — 110 BPM  (D)    ← İç→Dış halka ✓
```
Her geçiş 8 bar (≈15 saniye) crossfade ile yumuşatılır.

---

### 12.4 Maliyet – Gelir Analizi

#### Kurulum Maliyeti (Tek Seferlik)

| Kalem | Maliyet |
|-------|---------|
| Donanım (Raspberry Pi seti) | ₺2.720 |
| Teknisyen kurulum ziyareti (2 saat) | ₺500 |
| İlk müzik kütüphanesi üretimi (AI, ~500 parça) | ₺300 (bulut GPU) |
| **Toplam kurulum** | **₺3.520** |

#### Aylık Gider (TrackBang Tarafı)

| Kalem | Maliyet/ay |
|-------|-----------|
| Bulut sunucu payı (1 mekan) | ₺45 |
| AI müzik üretimi (aylık ~50 yeni parça) | ₺30 |
| Destek & bakım payı | ₺25 |
| **Toplam aylık gider** | **₺100** |

#### Gelir ve Karlılık

| Metrik | Değer |
|--------|-------|
| Aylık abonelik geliri | ₺399 |
| Aylık gider | ₺100 |
| **Brüt kar (aylık)** | **₺299 (%75 marj)** |
| Donanım geri ödeme süresi | ~12 ay (donanımı müşteri öder) |
| Abonelik geri ödeme süresi | **1 ay** |

> **Müşteri için karşılaştırma:**
> - Mevcut durum: Spotify Premium ₺149/ay (yasadışı ticari kullanım) + MESAM ceza riski (₺5.000–₺50.000)
> - TrackBang: ₺399/ay — tamamen yasal, daha iyi ses deneyimi, sıfır ceza riski

---

### 12.5 Pilot Başarı Kriterleri

**Teknik (1. Ay Sonu)**
- [ ] Sistem 30 gün boyunca kesintisiz çalıştı (uptime ≥ %98)
- [ ] Hiçbir müşteri şikayeti müzikle ilgili değildi
- [ ] İnternet kesintisinde çevrimdışı mod 4+ saat sorunsuz çalıştı
- [ ] Günlük enerji eğrisi program saatlerine %95 uydu

**İş (3. Ay Sonu)**
- [ ] Mekan sahibi aboneliği yeniledi (churn = 0)
- [ ] En az 1 referans müşteri getirdi
- [ ] Net Promoter Score (tek soruluk anket): ≥ 8/10
- [ ] "Müzik hakkında şikayetiniz azaldı mı?" → Evet

**Araştırma (3. Ay Sonu, Opsiyonel)**
- [ ] A/B haftaları karşılaştırması: TrackBang haftası vs. rastgele playlist haftası
- [ ] Kasa verisi: ortalama sipariş tutarı, mekanda kalma süresi
- [ ] Bu veri S1 araştırma sorusuna (Bölüm 7) ilk kanıtı sağlar

---

### 12.6 Risk ve Azaltma

| Risk | Olasılık | Etki | Azaltma |
|------|----------|------|---------|
| İnternet kesintisi | Orta | Yüksek | Raspberry Pi'da 12 saatlik yerel buffer |
| Mekan sahibi müziği beğenmedi | Orta | Yüksek | Profil anketi + 7 günlük ücretsiz ince ayar süreci |
| Ses sistemi uyumsuzluğu | Düşük | Orta | Kurulum öncesi 3.5mm / Bluetooth uyumluluk kontrolü |
| Raspberry Pi arızası | Düşük | Yüksek | Web player yedek mod (sadece tarayıcı + internet) |
| Müzik tekrarı şikayeti | Orta | Orta | Aylık 50 yeni parça ekleme, 30 günlük rotasyon havuzu |

---

### 12.7 Ölçekleme: 1 Mekandan 50 Mekana

Kadıköy pilot başarılı olursa aynı süreç şablon haline gelir:

```
1 Mekan (Ay 1-3):   Teknik doğrulama
      ↓
5 Mekan (Ay 4-6):   Operasyonel doğrulama (kurulum + destek ölçeği)
      ↓
20 Mekan (Ay 7-12): Birim ekonomisi doğrulama (CAC, LTV, churn)
      ↓
50 Mekan (Ay 13+):  Büyüme modu — referal kanalı + dijital pazarlama
```

**50 mekana ulaşıldığında:**
- Aylık Tekrarlayan Gelir (MRR): 50 × ₺399 = **₺19.950/ay**
- Aylık toplam gider: 50 × ₺100 = ₺5.000
- **Net kar: ₺14.950/ay (%75 marj)**
- 1 tam zamanlı teknik destek personeli karşılanabilir seviye

---

*Bu belge yaşayan bir araştırma notudur. Son güncelleme: Haziran 2026.*  
*Sonraki gözden geçirme: MVP ilk 5 beta mekanda konuşlandırıldığında.*
