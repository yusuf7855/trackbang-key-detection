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


def predict(mp3_path: str, model_path: str = DEFAULT_MODEL) -> dict:
    """
    Bir MP3 dosyasından key ve BPM tahmin eder.

    Döndürür
    --------
    dict:
        key        : str   — tahmin edilen ton (ör. 'A Minor')
        camelot    : str   — Camelot kodu (ör. '8A')
        confidence : float — softmax güven skoru [0-1]
        bpm        : float — tahmini BPM
        top3       : list  — [(key, camelot, confidence), ...] en olası 3 ton
    """
    import tensorflow as tf

    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"Dosya bulunamadı: {mp3_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model bulunamadı: {model_path}\n"
            "Önce eğitimi tamamla: python src/train.py"
        )

    # 1) Chroma CENS özelliği çıkar
    chroma = extract_all_features(mp3_path, target_length=TARGET_LENGTH)
    # (12, 1292) → (1, 12, 1292, 1)
    chroma_input = chroma[:, :TARGET_LENGTH, np.newaxis][np.newaxis, ...]

    # 2) Model yükle ve tahmin et
    model = tf.keras.models.load_model(model_path)
    key_probs, bpm_norm = model.predict(chroma_input, verbose=0)

    key_probs = key_probs[0]                          # (24,)
    bpm_val   = float(bpm_norm[0][0]) * 200.0         # normalize geri al

    # 3) Sonuçları hazırla
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
    }


def main():
    parser = argparse.ArgumentParser(description="Key & BPM Tahmini")
    parser.add_argument("mp3",    help="MP3 dosya yolu")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model dosyası (.keras)")
    args = parser.parse_args()

    print(f"\nDosya  : {os.path.basename(args.mp3)}")
    print("Analiz ediliyor...\n")

    result = predict(args.mp3, model_path=args.model)

    print("=" * 45)
    print(f"  Ton      : {result['key']:<14}  [{result['camelot']}]")
    print(f"  Guven    : %{result['confidence']*100:.1f}")
    print(f"  BPM      : {result['bpm']}")
    print("=" * 45)
    print("  Alternatifler:")
    for key, camelot, conf in result["top3"]:
        bar = "=" * int(conf * 30)
        print(f"    {key:<14} [{camelot}]  {bar}  %{conf*100:.1f}")
    print()


if __name__ == "__main__":
    main()
