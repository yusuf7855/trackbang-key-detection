"""
model.py — v2
Residual + Squeeze-and-Excitation Attention tabanlı Key Detection CNN.

v2 İyileştirmeleri (v1'e göre):
--------------------------------
1. Residual connections
   - Gradyanların derin katmanlara ulaşmasını kolaylaştırır (vanishing gradient önlenir)
   - Modelin "hiçbir şey öğrenmek yerine en azından atlama bağlantısını kullan" garantisi

2. Squeeze-and-Excitation (SE) Attention
   - Her konvolüsyon bloğundan sonra kanal bazlı dikkat
   - "Hangi pitch class'lar bu key için önemli?" sorusunu öğrenir
   - Örnek: G Major için G, B, D binleri ağırlıklı hale gelir

3. Leaky ReLU
   - Dying ReLU sorununu önler (negatif değerler tamamen sıfırlanmaz)

4. Daha geniş filtre sayıları (64 → 128 → 256 → 256)
   - Daha fazla özellik kapasitesi

5. BPM loss weight azaltıldı (0.1 → 0.05)
   - Key detection'a daha fazla odaklanır
"""

import tensorflow as tf
from tensorflow.keras import layers, Model


# 24 key sınıfı (Camelot sistemiyle eşleştirilmiş)
KEY_CLASSES = [
    'C Major',  'C Minor',
    'C# Major', 'C# Minor',
    'D Major',  'D Minor',
    'D# Major', 'D# Minor',
    'E Major',  'E Minor',
    'F Major',  'F Minor',
    'F# Major', 'F# Minor',
    'G Major',  'G Minor',
    'G# Major', 'G# Minor',
    'A Major',  'A Minor',
    'A# Major', 'A# Minor',
    'B Major',  'B Minor',
]

NUM_CLASSES = len(KEY_CLASSES)  # 24


# ── Yardımcı bloklar ─────────────────────────────────────────────────────────

def se_block(x, ratio=8):
    """
    Squeeze-and-Excitation bloğu — kanal bazlı dikkat mekanizması.

    Tasarım gerekçesi:
    ------------------
    Chroma matrisinde 12 kanal = 12 nota sınıfı.
    Bir akorun hangi notaları içerdiği key'i belirler (örn. G, B, D → G Major).
    SE bloğu "hangi notalar (kanallar) bu parça için daha önemli?" sorusunu öğrenir.

    Mimari:
      GAP → Dense(C/ratio, relu) → Dense(C, sigmoid) → kanal çarpımı

    ratio=8: 64 kanal → 8 nöron (yeterince kompakt, overfit yok)
    """
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)          # (B, C)
    se = layers.Reshape((1, 1, channels))(se)         # (B, 1, 1, C)
    se = layers.Dense(max(1, channels // ratio), activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    return layers.Multiply()([x, se])


def residual_block(x, filters, kernel_size=(3, 3)):
    """
    Residual block: Conv-BN-LeakyReLU-Conv-BN-SE + shortcut.

    Shortcut bağlantısı:
    - Girdi ve çıktı filtre sayısı aynıysa direkt toplama
    - Farklıysa 1×1 Conv ile boyut eşleştirme
    """
    shortcut = x

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # SE attention
    x = se_block(x, ratio=8)

    # Shortcut boyut eşleştirme
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.LeakyReLU(0.1)(x)
    return x


# ── Ana model ─────────────────────────────────────────────────────────────────

def build_model(input_shape=(12, 1292, 1), lr=0.001):
    """
    Residual + SE-Attention CNN — Key & BPM Detection.

    Giriş: (12, 1292, 1) — 12 chroma bin × 1292 zaman adımı
    Çıktı 1: (24,) softmax — key sınıfı
    Çıktı 2: (1,)  linear  — BPM (normalize)

    Mimari gerekçe:
    ---------------
    - Dikeysel (frekans) boyut = 12 → küçük, sınırlı havuzlama yapılır
    - Yataysal (zaman) boyut = 1292 → büyük, progresif havuzlama
    - Global Average Pooling: zamana göre öğrenilmiş bir "key profili" üretir
      → Krumhansl-Schmuckler algoritmasının öğrenen versiyonu
    """
    inputs = layers.Input(shape=input_shape, name='chroma_input')

    # İlk projeksiyon katmanı
    x = layers.Conv2D(64, (3, 7), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.1)(x)

    # ── Blok 1: Kaba zamansal özellikler (64 filtre) ─────────────
    x = residual_block(x, 64, kernel_size=(3, 7))
    x = layers.MaxPooling2D((1, 4))(x)   # zaman: 1292 → 323
    x = layers.Dropout(0.2)(x)

    # ── Blok 2: Orta seviye özellikler (128 filtre) ──────────────
    x = residual_block(x, 128, kernel_size=(3, 5))
    x = layers.MaxPooling2D((1, 4))(x)   # zaman: 323 → 80
    x = layers.Dropout(0.2)(x)

    # ── Blok 3: Yüksek seviye özellikler (256 filtre) ────────────
    x = residual_block(x, 256, kernel_size=(3, 3))
    x = layers.MaxPooling2D((1, 4))(x)   # zaman: 80 → 20
    x = layers.Dropout(0.2)(x)

    # ── Blok 4: En soyut özellikler (256 filtre) ─────────────────
    x = residual_block(x, 256, kernel_size=(3, 3))
    x = layers.Dropout(0.2)(x)

    # ── Global Average Pooling → öğrenilmiş key profili ──────────
    x = layers.GlobalAveragePooling2D()(x)

    # ── Ortak Dense katmanlar ─────────────────────────────────────
    shared = layers.Dense(512)(x)
    shared = layers.BatchNormalization()(shared)
    shared = layers.LeakyReLU(0.1)(shared)
    shared = layers.Dropout(0.5)(shared)

    shared = layers.Dense(256)(shared)
    shared = layers.BatchNormalization()(shared)
    shared = layers.LeakyReLU(0.1)(shared)
    shared = layers.Dropout(0.3)(shared)

    # ── Key çıktısı (24 sınıf softmax) ───────────────────────────
    key_output = layers.Dense(NUM_CLASSES, activation='softmax',
                               name='key_output')(shared)

    # ── BPM çıktısı (regresyon) ───────────────────────────────────
    bpm_output = layers.Dense(1, activation='linear',
                               name='bpm_output')(shared)

    model = Model(inputs=inputs, outputs=[key_output, bpm_output])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={
            'key_output': tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            'bpm_output': 'mse'
        },
        loss_weights={
            'key_output': 1.0,
            'bpm_output': 0.05,   # BPM'e daha az ağırlık → key'e odaklan
        },
        metrics={
            'key_output': 'accuracy',
            'bpm_output': 'mae'
        }
    )

    return model


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────

def key_to_index(key_str: str) -> int:
    """'A Minor' → 19"""
    try:
        return KEY_CLASSES.index(key_str)
    except ValueError:
        return -1


def index_to_key(idx: int) -> str:
    """19 → 'A Minor'"""
    return KEY_CLASSES[idx]


def camelot_to_key(camelot: str):
    """
    '8A' → 'A Minor', '8B' → 'C Major'
    Camelot → standart key dönüşümü
    """
    CAMELOT_MAP = {
        '1A': 'G# Minor', '1B': 'B Major',
        '2A': 'D# Minor', '2B': 'F# Major',
        '3A': 'A# Minor', '3B': 'C# Major',
        '4A': 'F Minor',  '4B': 'G# Major',
        '5A': 'C Minor',  '5B': 'D# Major',
        '6A': 'G Minor',  '6B': 'A# Major',
        '7A': 'D Minor',  '7B': 'F Major',
        '8A': 'A Minor',  '8B': 'C Major',
        '9A': 'E Minor',  '9B': 'G Major',
        '10A': 'B Minor', '10B': 'D Major',
        '11A': 'F# Minor','11B': 'A Major',
        '12A': 'C# Minor','12B': 'E Major',
    }
    return CAMELOT_MAP.get(camelot.upper(), None)


if __name__ == '__main__':
    model = build_model()
    model.summary(line_length=80)
    print(f"\nToplam key sınıfı : {NUM_CLASSES}")
    print(f"Toplam parametre  : {model.count_params():,}")
