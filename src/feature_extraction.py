"""
feature_extraction.py  — v2
MP3 dosyasından Chroma CENS özelliği çıkarır.

v2 İyileştirmesi: HPSS (Harmonic-Percussive Source Separation)
--------------------------------------------------------------
Key detection için en önemli bilgi pitch class dağılımıdır.
Müzikal ton (key), yalnızca harmonik bileşenden (akorlar, melodiler) gelir;
perküsif bileşen (davul, vurucu sesler) gürültü ekler ve chroma'yı kirletir.

HPSS ile harmonik bileşeni ayırarak çok daha temiz ve güvenilir
bir chroma matrisi elde edilir → accuracy artışı sağlar.

Chroma CENS (Energy Normalized Statistics):
  - 12 bin → 12 nota sınıfı (C, C#, D, ..., B)
  - Oktavdan bağımsız: aynı nota hangi oktavda çalınırsa çalınsın aynı bin
  - CENS versiyonu gürültüye ve dinamik değişimlere karşı dayanıklı
  - MIR literatüründe key detection için standart özellik

Pitch shift augmentasyonu CHROMA üzerinde matematiksel olarak doğrudur:
  np.roll(chroma, n, axis=0) → tam olarak n yarım ton yukarı kaydırma
"""

import librosa
import numpy as np

# Sabitler
SAMPLE_RATE = 22050
DURATION    = 30       # Kullanılacak süre (saniye)
HOP_LENGTH  = 512
N_FFT       = 2048
N_CHROMA    = 12       # Nota sınıfı sayısı


def load_audio(file_path: str, duration: float = DURATION) -> tuple:
    """MP3/WAV dosyasını mono olarak yükler."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=duration, mono=True)
    return y, sr


def extract_chroma_cens(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Harmonik bileşenden Chroma CENS çıkarır.

    Neden HPSS?
    -----------
    Perküsif sesler (kick, snare, hi-hat) chroma matrisine gürültü katar.
    Harmonik bileşen ayrılırsa model yalnızca tonal bilgiyi öğrenir.
    Bu, key detection accuracy'sini anlamlı biçimde artırır.

    Döndürür
    --------
    chroma : (12, T) — değerler [0, 1] aralığında
    """
    # HPSS: Harmonik bileşeni perküsif bileşenden ayır
    # margin=3.0 → harmonik/perküsif ayrımını daha agresif yapar
    y_harmonic, _ = librosa.effects.hpss(y, margin=3.0)

    chroma = librosa.feature.chroma_cens(
        y=y_harmonic,
        sr=sr,
        hop_length=HOP_LENGTH,
        n_chroma=N_CHROMA,
    )
    return chroma


def extract_all_features(file_path: str, target_length: int = 1292) -> np.ndarray:
    """
    MP3 dosyasından normalize edilmiş Chroma CENS (harmonik) çıkarır.

    Parametreler
    ------------
    file_path    : MP3 dosya yolu
    target_length: zaman boyutu (30sn @ 22050Hz / 512 hop ≈ 1292 frame)

    Döndürür
    --------
    chroma : np.ndarray, shape (12, target_length), dtype float32
    """
    try:
        y, sr = load_audio(file_path)
    except Exception:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)

    chroma = extract_chroma_cens(y, sr)   # (12, T)

    # Sabit uzunluğa getir (kırp veya sıfır-doldur)
    if chroma.shape[1] >= target_length:
        chroma = chroma[:, :target_length]
    else:
        pad = target_length - chroma.shape[1]
        chroma = np.pad(chroma, ((0, 0), (0, pad)), mode='constant')

    # Z-score normalizasyonu — her dosyayı kendi istatistikleriyle normalize et
    chroma = (chroma - chroma.mean()) / (chroma.std() + 1e-8)

    return chroma.astype(np.float32)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Dosya: {path}")
        chroma = extract_all_features(path)
        print(f"Chroma CENS shape : {chroma.shape}")
        print(f"Min: {chroma.min():.3f}  Max: {chroma.max():.3f}  Mean: {chroma.mean():.3f}")
    else:
        print("Kullanım: python feature_extraction.py <mp3_dosyası>")
