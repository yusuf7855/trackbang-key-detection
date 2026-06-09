# Proje Fikri — TrackBang Audio Intelligence

## Sorun

DJ'ler set hazırlarken parçaları **müzikal ton (key)** ve **BPM** uyumuna göre seçer. Bu süreç şu an büyük ölçüde manüeldir:

- Beatport, Spotify gibi platformlar key/BPM bilgisini her zaman sağlamaz
- Sağlansa bile güvenilirliği tartışmalıdır
- TrackBang platformuna yüklenen yeni şarkıların otomatik analizi için harici API'lere bağımlılık pahalı ve kısıtlayıcıdır

## Fikir

**30 saniyelik MP3 önizlemesinden**, yalnızca ses sinyalini kullanarak:

1. Müzikal tonu (24 sınıf: 12 Majör + 12 Minör) tahmin et
2. BPM değerini hesapla

...yapabilen bir model eğit. Böylece TrackBang'deki her yeni parça otomatik olarak etiketlenebilir.

## Neden Chroma CENS?

Müzikal ton, frekans içeriğiyle doğrudan ilişkilidir. **Chroma** özelliği, 12 yarım-ton sınıfındaki enerji dağılımını yakalar. CENS (Chroma Energy Normalized Statistics) versiyonu gürültüye karşı daha dayanıklıdır.

## Neden Derin Öğrenme?

Geleneksel yöntemler (örn. NNLS Chroma, Krumhansl-Schmuckler algoritması) sabit kurallara dayanır ve elektronik müzikteki karmaşık tonal yapılara uymaz. CNN tabanlı bir model, veri üzerinden örüntüyü öğrenebilir.

## TrackBang ile Bütünleşme

```
Kullanıcı → "8A tonunda 127 BPM peaktime set yap"
                    ↓
            CUE AI (Groq LLM)
                    ↓
        Bu modelin ürettiği key/BPM etiketleri
                    ↓
        Camelot uyumluluk + BPM eşleştirme
                    ↓
            10 parçalık harmonik set
```

Model olmadan CUE AI, sadece manüel etiketlenmiş şarkılara erişebilir. Model ile her yeni yükleme otomatik olarak analiz edilir.

## Başlangıç Hipotezi

- Elektronik müzikte chroma özelliği key için yeterince ayırt edici bilgi içerir
- HPSS ile perküsif seslerin temizlenmesi doğruluğu artırır
- Residual CNN + SE Attention yapısı, basit CNN'den daha iyi genelleme sağlar
- Çoklu görev öğrenimi (key + BPM aynı anda) parametre verimliliği sağlar

## Özgünlük

- Piyasadaki araçlar (Essentia, Librosa key_detection) genel müzik için tasarlanmış
- Bu model **yalnızca elektronik müzik** üzerinde eğitildi: daha odaklı, daha doğru
- TrackBang'in gerçek üretim veritabanından gelen etiketler kullanıldı
