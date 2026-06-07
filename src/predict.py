"""
predict.py
==========
Eğitilmiş modeli kullanarak bir MP3 dosyasının müzikal tonunu ve BPM'ini tahmin eder.

Kullanım:
  python src/predict.py <mp3_dosyası>
  python src/predict.py data/raw/previews/ornek.mp3
  python src/predict.py ornek.mp3 --model models/key_detection_model.keras
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='Custom')
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr=0.001, warmup_steps=545, total_steps=13080, min_lr=1e-6, **kw):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = float(warmup_steps)
        self.total_steps  = float(total_steps)
        self.min_lr       = min_lr
    def __call__(self, step):
        step     = tf.cast(step, tf.float32)
        warmup   = self.peak_lr * (step / self.warmup_steps)
        cos_lr   = self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (
            1 + tf.cos(3.14159265 * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
        return tf.where(step < self.warmup_steps, warmup, cos_lr)
    def get_config(self):
        return {'peak_lr': self.peak_lr, 'warmup_steps': self.warmup_steps,
                'total_steps': self.total_steps, 'min_lr': self.min_lr}

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extraction import extract_all_features
from model import KEY_CLASSES, index_to_key

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "models", "key_detection_model.keras")

TARGET_LENGTH = 1292
N_CHROMA      = 12

KEY_TO_CAMELOT = {
    'C Major': '8B',  'C Minor': '5A',
    'C# Major': '3B', 'C# Minor': '12A',
    'D Major': '10B', 'D Minor': '7A',
    'D# Major': '5B', 'D# Minor': '2A',
    'E Major': '12B', 'E Minor': '9A',
    'F Major': '7B',  'F Minor': '4A',
    'F# Major': '2B', 'F# Minor': '11A',
    'G Major': '9B',  'G Minor': '6A',
    'G# Major': '4B', 'G# Minor': '1A',
    'A Major': '11B', 'A Minor': '8A',
    'A# Major': '6B', 'A# Minor': '3A',
    'B Major': '1B',  'B Minor': '10A',
}


def _tta_predict(model, chroma: np.ndarray, max_shift: int = 3) -> np.ndarray:
    """
    Test-Time Augmentation: eğitim dağılımındaki shift'lerle tahmin ortalaması.

    max_shift: eğitim sırasında kullanılan pitch shift aralığı ile aynı olmalı.
    Dağılım dışı shift'ler (±4, ±5...) modeli yanıltır → sadece ±max_shift kullanılır.

    Shifts: -max_shift, ..., 0, ..., +max_shift → 2*max_shift+1 versiyon
    """
    NUM_CLASSES = 24
    shifts = list(range(-max_shift, max_shift + 1))   # örn. [-3,-2,-1,0,1,2,3]
    n = len(shifts)
    batch = np.zeros((n, N_CHROMA, TARGET_LENGTH, 1), dtype=np.float32)

    for i, s in enumerate(shifts):
        shifted = np.roll(chroma, s, axis=0)
        batch[i] = shifted[:, :TARGET_LENGTH, np.newaxis]

    key_preds, bpm_preds = model.predict(batch, verbose=0)

    # Her tahmini orijinal key uzayına geri-kaydır
    aggregated = np.zeros(NUM_CLASSES, dtype=np.float32)
    for i, s in enumerate(shifts):
        probs = key_preds[i]
        unshifted = np.zeros(NUM_CLASSES, dtype=np.float32)
        for pred_idx in range(NUM_CLASSES):
            orig_idx = (pred_idx - s * 2) % NUM_CLASSES
            unshifted[orig_idx] += probs[pred_idx]
        aggregated += unshifted

    aggregated /= n
    bpm_avg = float(np.mean(bpm_preds)) * 200.0
    return aggregated, bpm_avg


def predict(mp3_path: str, model_path: str = DEFAULT_MODEL, tta: bool = False) -> dict:
    """
    Bir MP3 dosyasından key ve BPM tahmin eder.

    Parametreler
    ------------
    tta : bool
        Test-Time Augmentation kullan (varsayılan: True)
        12 pitch shift ortalaması → +3-5% doğruluk

    Döndürür
    --------
    dict:
        key        : str   — tahmin edilen ton (ör. 'A Minor')
        camelot    : str   — Camelot kodu (ör. '8A')
        confidence : float — softmax güven skoru [0-1]
        bpm        : float — tahmini BPM
        top3       : list  — [(key, camelot, confidence), ...] en olası 3 ton
        tta        : bool  — TTA kullanıldı mı
    """
    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"Dosya bulunamadı: {mp3_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model bulunamadı: {model_path}\n"
            "Önce eğitimi tamamla: python src/train.py"
        )

    # 1) Chroma CENS özelliği çıkar
    chroma = extract_all_features(mp3_path, target_length=TARGET_LENGTH)  # (12, 1292)

    # 2) Model yükle
    model = tf.keras.models.load_model(model_path, compile=False)

    # 3) Tahmin — TTA veya tek geçiş
    if tta:
        key_probs, bpm_val = _tta_predict(model, chroma, max_shift=3)
    else:
        chroma_input = chroma[:, :TARGET_LENGTH, np.newaxis][np.newaxis, ...]
        kp, bn = model.predict(chroma_input, verbose=0)
        key_probs = kp[0]
        bpm_val   = float(bn[0][0]) * 200.0

    # 4) Sonuçları hazırla
    top_idx       = np.argsort(key_probs)[::-1]
    predicted_key = KEY_CLASSES[top_idx[0]]
    confidence    = float(key_probs[top_idx[0]])

    top3 = [
        (KEY_CLASSES[i], KEY_TO_CAMELOT.get(KEY_CLASSES[i], '?'), float(key_probs[i]))
        for i in top_idx[:3]
    ]

    return {
        "key":        predicted_key,
        "camelot":    KEY_TO_CAMELOT.get(predicted_key, "?"),
        "confidence": confidence,
        "bpm":        round(max(60.0, min(220.0, bpm_val)), 1),
        "top3":       top3,
        "tta":        tta,
    }


def main():
    parser = argparse.ArgumentParser(description="Key & BPM Tahmini")
    parser.add_argument("mp3",     help="MP3 dosya yolu")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model dosyası (.keras)")
    parser.add_argument("--no-tta", action="store_true", help="TTA'yı devre dışı bırak")
    args = parser.parse_args()

    print(f"\nDosya  : {os.path.basename(args.mp3)}")
    tta = not args.no_tta
    print(f"Mod    : {'TTA (12 pitch shift ortalaması)' if tta else 'Tek geçiş'}")
    print("Analiz ediliyor...\n")

    result = predict(args.mp3, model_path=args.model, tta=tta)

    print("=" * 45)
    print(f"  Ton      : {result['key']:<14}  [{result['camelot']}]")
    print(f"  Güven    : %{result['confidence']*100:.1f}")
    print(f"  BPM      : {result['bpm']}")
    print("=" * 45)
    print("  Alternatifler:")
    for key, camelot, conf in result["top3"]:
        bar = "=" * int(conf * 30)
        print(f"    {key:<14} [{camelot}]  {bar}  %{conf*100:.1f}")
    print()


if __name__ == "__main__":
    main()
