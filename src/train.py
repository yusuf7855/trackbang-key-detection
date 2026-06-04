"""
train.py — v2
tf.data generator kullanarak düşük RAM'de CNN eğitimi.

v2 İyileştirmeleri:
--------------------
1. Sınıf ağırlıkları (class weights)
   - Veri setinde bazı key'ler daha az temsil ediliyor (G Minor: 275, C# Minor: 83)
   - Nadir sınıflara daha yüksek loss ağırlığı vererek dengesizlik giderilir

2. Cosine Annealing öğrenme hızı
   - Sabit LR yerine cosine eğrisi ile azalan LR
   - Daha iyi sonuçlara yakınsama sağlar

3. Frekans maskeleme augmentasyonu (SpecAugment tamamlayıcısı)
   - Zaman maskelemesine ek olarak chroma binlerini de maskeler
   - Modelin belirli notaların yokluğuna karşı dayanıklı olmasını sağlar

4. --recache bayrağı
   - feature_extraction.py değiştiğinde cache'i yeniden oluşturmak için

Kullanım:
  python src/train.py
  python src/train.py --epochs 100 --batch 16
  python src/train.py --recache      # feature extraction değişince
"""

import os
import sys
import argparse
import csv
import random
import numpy as np
import tensorflow as tf
from typing import List
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extraction import extract_all_features
from model import build_model, KEY_CLASSES, key_to_index

# ── Sabitler ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREVIEW_DIR  = os.path.join(PROJECT_ROOT, "data", "raw", "previews")
LABELS_CSV   = os.path.join(PROJECT_ROOT, "data", "labels.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
CACHE_DIR    = os.path.join(PROJECT_ROOT, "data", "processed", "cache")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(CACHE_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGET_LENGTH = 1292
N_CHROMA      = 12
NUM_CLASSES   = len(KEY_CLASSES)


# ── Etiket yükleme ─────────────────────────────────────────────────────────────

def load_labels(csv_path: str) -> List[dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = key_to_index(row["musical_key"])
            if idx == -1:
                continue
            bpm = float(row["bpm"]) if row.get("bpm") else 120.0
            if bpm <= 0:
                bpm = 120.0
            audio_path = os.path.join(PREVIEW_DIR, row["filename"])
            rows.append({
                "path":    audio_path,
                "key_idx": idx,
                "bpm":     bpm / 200.0,
            })
    return rows


# ── Cache yönetimi ─────────────────────────────────────────────────────────────

def get_cache_path(audio_path: str) -> str:
    name = os.path.splitext(os.path.basename(audio_path))[0]
    return os.path.join(CACHE_DIR, f"{name}.npy")


def ensure_cache(rows: List[dict], recache: bool = False):
    """
    Eksik (veya --recache ile tüm) cache dosyalarını oluştur.
    feature_extraction.py değiştiğinde --recache kullan.
    """
    if recache:
        print("--recache aktif: tüm cache yeniden oluşturuluyor...")
        to_process = rows
    else:
        to_process = [r for r in rows if not os.path.exists(get_cache_path(r["path"]))]

    if not to_process:
        print(f"Cache hazır ({len(rows)} dosya).")
        return

    print(f"\n{len(to_process)} dosya için Chroma CENS (HPSS) cache oluşturuluyor...")
    failed = 0
    for row in tqdm(to_process, desc="Cache", unit="dosya"):
        try:
            chroma = extract_all_features(row["path"], target_length=TARGET_LENGTH)
            np.save(get_cache_path(row["path"]), chroma)
        except Exception as e:
            failed += 1
    if failed:
        print(f"Cache tamamlandı. Başarısız: {failed}")
    else:
        print(f"Cache tamamlandı. Tüm dosyalar başarıyla işlendi.")


# ── Sınıf ağırlıkları ──────────────────────────────────────────────────────────

def compute_class_weights(rows: List[dict]) -> dict:
    """
    Sınıf dengesizliğini gidermek için inverse-frequency ağırlıkları hesaplar.

    Formül: w_i = N_total / (N_classes * count_i)
    - Nadir sınıflar yüksek ağırlık → modelin onlara daha fazla dikkat etmesi
    - Bol sınıflar düşük ağırlık

    Örnek:
      G Minor: 275 örnek → düşük ağırlık
      C# Minor: 83 örnek → yüksek ağırlık
    """
    counts = Counter(r["key_idx"] for r in rows)
    total  = sum(counts.values())
    n_cls  = NUM_CLASSES

    weights = {}
    for cls_idx in range(n_cls):
        count = counts.get(cls_idx, 1)
        weights[cls_idx] = total / (n_cls * count)

    # Normalize: max ağırlık 5'i geçmesin (aşırı ağırlık instability yaratır)
    max_w = max(weights.values())
    if max_w > 5.0:
        scale = 5.0 / max_w
        weights = {k: v * scale for k, v in weights.items()}

    print("\nSınıf ağırlıkları (ilk 5):")
    for i, (k, w) in enumerate(sorted(weights.items(), key=lambda x: -x[1])[:5]):
        print(f"  {KEY_CLASSES[k]:<14}: {w:.3f}")

    return {"key_output": weights}


# ── Augmentasyon ───────────────────────────────────────────────────────────────

def augment_fn(chroma, labels):
    """
    Chroma CENS üzerinde çok katmanlı augmentasyon.

    1. Pitch shift (±5 semitone)  — CHROMA'da matematiksel olarak TAM doğru
    2. Time masking               — SpecAugment (zaman dilimi sıfırlama)
    3. Frequency masking          — SpecAugment (chroma bin sıfırlama)
    4. Gaussian gürültü           — küçük rastgele gürültü
    """
    key_oh  = labels["key_output"]
    bpm_val = labels["bpm_output"]

    # ── 1. Pitch shift ────────────────────────────────────────────
    # 1 chroma bin = tam 1 yarım ton → np.roll matematiksel olarak doğru
    semitone    = tf.random.uniform([], minval=-5, maxval=6, dtype=tf.int32)
    chroma      = tf.roll(chroma, semitone, axis=0)
    key_idx     = tf.cast(tf.argmax(key_oh), tf.int32)
    new_key_idx = tf.math.floormod(key_idx + semitone * 2, NUM_CLASSES)
    key_oh      = tf.one_hot(new_key_idx, NUM_CLASSES)

    # ── 2. Time masking ───────────────────────────────────────────
    t_width = tf.random.uniform([], 0, 100, tf.int32)
    t_start = tf.random.uniform([], 0, tf.maximum(1, TARGET_LENGTH - t_width), tf.int32)
    time_idx  = tf.range(TARGET_LENGTH)
    time_keep = tf.cast(
        tf.logical_or(time_idx < t_start, time_idx >= t_start + t_width),
        tf.float32
    )
    chroma = chroma * time_keep[tf.newaxis, :, tf.newaxis]

    # ── 3. Frequency masking (yeni) ───────────────────────────────
    # Chroma binlerini maskele: bazı nota sınıflarının yokluğuna karşı dayanıklılık
    f_width = tf.random.uniform([], 0, 3, tf.int32)   # en fazla 3 bin maskele
    f_start = tf.random.uniform([], 0, tf.maximum(1, N_CHROMA - f_width), tf.int32)
    freq_idx  = tf.range(N_CHROMA)
    freq_keep = tf.cast(
        tf.logical_or(freq_idx < f_start, freq_idx >= f_start + f_width),
        tf.float32
    )
    chroma = chroma * freq_keep[:, tf.newaxis, tf.newaxis]

    # ── 4. Gaussian gürültü ───────────────────────────────────────
    noise  = tf.random.normal(tf.shape(chroma), mean=0.0, stddev=0.02)
    chroma = chroma + noise

    return chroma, {"key_output": key_oh, "bpm_output": bpm_val}


# ── tf.data Dataset ────────────────────────────────────────────────────────────

def make_dataset(rows: List[dict], batch_size: int, shuffle: bool,
                 augment: bool = False) -> tf.data.Dataset:
    paths    = [get_cache_path(r["path"]) for r in rows]
    key_idxs = [r["key_idx"] for r in rows]
    bpms     = [r["bpm"] for r in rows]

    def load_sample(path_bytes, key_idx, bpm):
        path   = path_bytes.numpy().decode("utf-8")
        chroma = np.load(path).astype(np.float32)
        chroma = chroma[:, :TARGET_LENGTH, np.newaxis]   # (12, 1292, 1)
        key_one_hot = np.zeros(NUM_CLASSES, dtype=np.float32)
        key_one_hot[key_idx] = 1.0
        return chroma, key_one_hot, np.float32(bpm)

    def tf_load(path, key_idx, bpm):
        chroma, key_oh, bpm_val = tf.py_function(
            load_sample, [path, key_idx, bpm],
            [tf.float32, tf.float32, tf.float32]
        )
        chroma.set_shape([N_CHROMA, TARGET_LENGTH, 1])
        key_oh.set_shape([NUM_CLASSES])
        bpm_val.set_shape([])
        return chroma, {"key_output": key_oh, "bpm_output": bpm_val}

    ds = tf.data.Dataset.from_tensor_slices((paths, key_idxs, bpms))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(rows), reshuffle_each_iteration=True)
    ds = ds.map(tf_load, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ── Ana Fonksiyon ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Key Detection CNN v2 Eğitimi")
    parser.add_argument("--epochs",   type=int,   default=120)
    parser.add_argument("--batch",    type=int,   default=16)
    parser.add_argument("--lr",       type=float, default=0.001)
    parser.add_argument("--val",      type=float, default=0.15)
    parser.add_argument("--recache",  action="store_true",
                        help="Feature extraction değiştiğinde cache'i yeniden oluştur")
    args = parser.parse_args()

    print("=" * 65)
    print("Key Detection CNN v2 — Residual + SE + Sınıf Ağırlıkları")
    print("=" * 65)
    print(f"  Epochs: {args.epochs} | Batch: {args.batch} | LR: {args.lr}")

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    # 1) Etiketler
    if not os.path.exists(LABELS_CSV):
        print(f"[HATA] {LABELS_CSV} yok. Önce build_dataset.py çalıştır.")
        sys.exit(1)

    rows = load_labels(LABELS_CSV)
    print(f"\n{len(rows)} örnek yüklendi.")

    # 2) Cache
    ensure_cache(rows, recache=args.recache)
    rows = [r for r in rows if os.path.exists(get_cache_path(r["path"]))]
    print(f"Cache'te hazır: {len(rows)} örnek")

    # 3) Train / Val split
    db_rows = [r for r in rows if not os.path.basename(r["path"]).startswith("sc_")]
    sc_rows = [r for r in rows if os.path.basename(r["path"]).startswith("sc_")]

    key_counts = Counter(r["key_idx"] for r in db_rows)
    db_rows    = [r for r in db_rows if key_counts[r["key_idx"]] >= 2]

    db_tr, val_rows = train_test_split(
        db_rows, test_size=args.val, random_state=42,
        stratify=[r["key_idx"] for r in db_rows]
    )
    tr_rows = db_tr + sc_rows
    random.shuffle(tr_rows)

    print(f"Train: {len(tr_rows)} (DB: {len(db_tr)} + Scraper: {len(sc_rows)}) | "
          f"Val: {len(val_rows)} (sadece DB)")

    # 4) Sınıf ağırlıkları
    class_weights = compute_class_weights(tr_rows)

    # 5) Dataset
    train_ds = make_dataset(tr_rows,  batch_size=args.batch, shuffle=True,  augment=True)
    val_ds   = make_dataset(val_rows, batch_size=args.batch, shuffle=False, augment=False)

    # 6) Model + Cosine Decay LR
    # Cosine Decay: LR başlangıçtan sıfıra kadar cosine eğrisi ile azalır
    # Her epoch başında LR yeniden hesaplanır → son epoch'larda çok küçük adımlar
    total_steps = len(tr_rows) // args.batch * args.epochs
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr,
        decay_steps=total_steps,
        alpha=1e-6,   # minimum LR
    )
    model = build_model(input_shape=(N_CHROMA, TARGET_LENGTH, 1), lr=lr_schedule)
    model.summary(line_length=70)

    # 7) Callback'ler
    model_path = os.path.join(MODELS_DIR, "key_detection_model.keras")
    callbacks  = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_key_output_accuracy", mode="max",
            save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_key_output_accuracy", mode="max",
            patience=20,                   # v1'de 15 idi, daha uzun bekle
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(RESULTS_DIR, "training_log.csv")
        ),
    ]

    # 8) Eğitim
    print(f"\nEğitim başlıyor → {model_path}\n")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # 9) Sonuç
    best_acc = max(history.history.get("val_key_output_accuracy", [0]))
    best_mae = min(history.history.get("val_bpm_output_mae", [999]))
    print("\n" + "=" * 65)
    print(f"Eğitim bitti!")
    print(f"  Val key accuracy : %{best_acc * 100:.1f}")
    print(f"  Val BPM MAE      : {best_mae * 200:.1f} BPM")
    print(f"  Model            : {model_path}")
    print("=" * 65)

    # H5 formatında da kaydet
    h5_path = os.path.join(MODELS_DIR, "key_detection_model.h5")
    model.save(h5_path)
    print(f"H5 formatı da kaydedildi: {h5_path}")


if __name__ == "__main__":
    main()
