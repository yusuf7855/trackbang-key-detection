"""
scraper.py
==========
SoundCloud'dan genre bazlı track indirir, key/BPM çıkarır, Chroma cache alır,
MP3'ü siler. Her RETRAIN_EVERY yeni örnekte modeli yeniden eğitir.

Key çıkarma önceliği:
  1. Başlık / description'da Camelot kodu  (8A, 10B, 1A–12B)
  2. Başlık / description'da müzikal key    (Am, F#m, A Minor, ...)
  3. Track tags                             (aynı parse mantığı)
  4. Hiçbiri yoksa → atla (etiket yok, eğitime katkısı olmaz)

BPM:
  - librosa beat tracker — elektronik müzik için %90+ doğruluk

Kullanım:
  python src/scraper.py
  python src/scraper.py --genres "afro house" "indie dance" --per-genre 500
  python src/scraper.py --genres "melodic techno" --per-genre 2000 --workers 3
"""

import os
import sys
import re
import csv
import json
import time
import random
import argparse
import tempfile
import subprocess
from typing import Optional

import numpy as np
import librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extraction import extract_all_features, SAMPLE_RATE, DURATION
from model import KEY_CLASSES, key_to_index

# ── Sabitler ──────────────────────────────────────────────────────────────────

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREVIEW_DIR   = os.path.join(PROJECT_ROOT, "data", "raw", "previews")
CACHE_DIR     = os.path.join(PROJECT_ROOT, "data", "processed", "cache")
LABELS_CSV    = os.path.join(PROJECT_ROOT, "data", "labels.csv")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
TRAIN_SCRIPT  = os.path.join(PROJECT_ROOT, "src", "train.py")
YTDLP         = os.path.join(PROJECT_ROOT, "venv", "bin", "yt-dlp")
FFMPEG        = "ffmpeg"

RETRAIN_EVERY = 500     # Her N yeni örnekte bir yeniden eğit
CLIP_DURATION = 30      # Saniye
TARGET_LENGTH = 1292

os.makedirs(PREVIEW_DIR, exist_ok=True)
os.makedirs(CACHE_DIR,   exist_ok=True)

# ── Varsayılan genre listesi ──────────────────────────────────────────────────

DEFAULT_GENRES = [
    "afro house",
    "indie dance",
    "melodic techno",
    "melodic house",
    "progressive house",
    "tech house",
    "deep house",
    "organic house",
    "minimal deep tech",
    "drum and bass",
    "liquid drum and bass",
    "psytrance",
    "trance",
    "ambient",
    "downtempo",
    "nu disco",
    "funky house",
    "soulful house",
    "tropical house",
    "future house",
]

# ── Camelot ↔ Key dönüşümü ───────────────────────────────────────────────────

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

ENHARMONIC = {
    'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
    'db': 'C#', 'eb': 'D#', 'gb': 'F#', 'ab': 'G#', 'bb': 'A#',
}

VALID_KEYS = set(KEY_CLASSES)

# ── Key parse ─────────────────────────────────────────────────────────────────

def _normalize_note(note: str) -> str:
    note = note.strip()
    for flat, sharp in ENHARMONIC.items():
        if note.startswith(flat) and (len(note) == len(flat) or not note[len(flat)].isalpha()):
            note = sharp + note[len(flat):]
            break
    return note[0].upper() + note[1:]


# DJ bağlamını gösteren anahtar kelimeler
_DJ_CONTEXT = re.compile(
    r'\b(bpm|key|camelot|house|techno|trance|dance|electronic|edm|dj|'
    r'remix|mix|track|beat|music|afro|indie|deep|melodic|progressive|'
    r'minimal|organic|psytrance|dnb|drum|bass|disco|funk|soul)\b',
    re.IGNORECASE
)

# Camelot kodunu DJ bağlamında doğrular
# Kabul: "[8A]", "(8A)", "- 8A", "| 8A", "8A -", "8A |", "8A 128bpm", başta/sonda
_CAMELOT_DJ_PATTERN = re.compile(
    r'(?:[\[\(|\-\s]|^)'          # önce: [, (, |, -, boşluk veya başlangıç
    r'#?'                          # isteğe bağlı # işareti (örn: #6A)
    r'(1[0-2]|[1-9])[AB]'         # Camelot kodu
    r'(?:[\]\)|\-\s\/]|$|\d)',     # sonra: ], ), |, -, boşluk, /, son veya rakam
    re.IGNORECASE
)


def parse_key(text: str) -> Optional[str]:
    """
    Metin içinden müzikal key çıkarır.
    Öncelik: Camelot kodu (DJ bağlamında) → yazılı key → kısa notasyon
    """
    if not text:
        return None

    # 1) Camelot: sadece DJ bağlamında (parantez, tire, pipe ile çevrilmiş)
    m = _CAMELOT_DJ_PATTERN.search(text)
    if m:
        # Tam Camelot kodunu bul
        m2 = re.search(r'(1[0-2]|[1-9])[AB]', m.group(0), re.IGNORECASE)
        if m2:
            code = m2.group(0).upper()
            key = CAMELOT_MAP.get(code)
            if key:
                return key

    # 2) Yazılı biçim: "A Minor", "F# Major", "Bb minor", "A min", "F maj"
    pattern_long = r'\b([A-Ga-g][#b]?)\s*(Major|Minor|maj|min)\b'
    m = re.search(pattern_long, text, re.IGNORECASE)
    if m:
        note = _normalize_note(m.group(1))
        mode = "Major" if m.group(2).lower().startswith("ma") else "Minor"
        key = f"{note} {mode}"
        if key in VALID_KEYS:
            return key

    # 3) Kısa notasyon: Am, F#m, Gmaj, Bb
    pattern_short = r'\b([A-Ga-g][#b]?)(m|maj)\b'
    m = re.search(pattern_short, text)
    if m:
        note = _normalize_note(m.group(1))
        mode = "Minor" if m.group(2) == "m" else "Major"
        key = f"{note} {mode}"
        if key in VALID_KEYS:
            return key

    return None


def is_dj_track(meta: dict) -> bool:
    """
    Track'in gerçek DJ/elektronik müzik olup olmadığını kontrol eder.
    Yanlış etiketlenmiş içerikleri (okul sınıfı '5B', oyun müziği vb.) filtreler.
    """
    text = " ".join([
        meta.get("title") or "",
        (meta.get("description") or "")[:200],
        " ".join(meta.get("tags") or []),
        " ".join(meta.get("genres") or []),
    ])
    return bool(_DJ_CONTEXT.search(text))


def extract_key_from_metadata(meta: dict) -> Optional[str]:
    """Metadata dict'ten key çıkarmayı dener."""
    # Önce SoundCloud'un kendi key_signature alanı
    if meta.get("key_signature"):
        k = parse_key(meta["key_signature"])
        if k:
            return k

    # Başlık
    k = parse_key(meta.get("title", ""))
    if k:
        return k

    # Description
    k = parse_key(meta.get("description", ""))
    if k:
        return k

    # Tags
    for tag in (meta.get("tags") or []):
        k = parse_key(tag)
        if k:
            return k

    return None


# ── BPM çıkarma ───────────────────────────────────────────────────────────────

def extract_bpm(y: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """librosa beat tracker ile BPM tahmin eder."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])
    return max(60.0, min(220.0, float(tempo)))


# ── SoundCloud tarama ─────────────────────────────────────────────────────────

def scrape_soundcloud_query(query: str, n: int = 50) -> list:
    """
    yt-dlp ile SoundCloud'dan arama sonuçlarını çeker.
    Her track için dict: {url, title, description, tags, key_signature, bpm}
    """
    search_query = f"scsearch{n}:{query}"
    cmd = [
        YTDLP,
        "--dump-json",
        "--flat-playlist",
        "--no-warnings",
        "--quiet",
        search_query,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print(f"  [!] Timeout: {genre}")
        return []

    tracks = []
    for line in result.stdout.strip().splitlines():
        try:
            d = json.loads(line)
            tracks.append({
                "url":           d.get("webpage_url") or d.get("url", ""),
                "sc_id":         str(d.get("id", "")),
                "title":         d.get("title", ""),
                "description":   d.get("description", ""),
                "tags":          d.get("tags") or [],
                "key_signature": d.get("key_signature"),
                "sc_bpm":        d.get("bpm"),
            })
        except Exception:
            continue
    return tracks


# ── İndirme ───────────────────────────────────────────────────────────────────

def download_clip(url: str, dest_path: str) -> bool:
    """SoundCloud URL'sinden 30 sn MP3 indirir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_tmpl = os.path.join(tmpdir, "audio.%(ext)s")
        dl_cmd = [
            YTDLP, url,
            "-x", "--audio-format", "mp3", "--audio-quality", "5",
            "-o", tmp_tmpl,
            "--quiet", "--no-warnings", "--no-playlist",
        ]
        try:
            r = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                return False
        except subprocess.TimeoutExpired:
            return False

        tmp_files = [f for f in os.listdir(tmpdir) if f.startswith("audio.")]
        if not tmp_files:
            return False
        tmp_audio = os.path.join(tmpdir, tmp_files[0])

        clip_cmd = [
            FFMPEG, "-y",
            "-i", tmp_audio,
            "-ss", "0",
            "-t",  str(CLIP_DURATION),
            "-acodec", "libmp3lame", "-q:a", "5",
            dest_path,
            "-loglevel", "error",
        ]
        try:
            r = subprocess.run(clip_cmd, capture_output=True, text=True, timeout=60)
            return r.returncode == 0 and os.path.exists(dest_path) and os.path.getsize(dest_path) > 10_000
        except subprocess.TimeoutExpired:
            return False


# ── Labels CSV ────────────────────────────────────────────────────────────────

def load_existing_labels() -> dict:
    """Mevcut labels.csv'den {filename: row} dict döndürür."""
    existing = {}
    if not os.path.exists(LABELS_CSV):
        return existing
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            existing[row["filename"]] = row
    return existing


def append_to_csv(rows: list):
    """Yeni satırları labels.csv'ye ekler (yoksa oluşturur)."""
    fieldnames = ["song_id", "filename", "musical_key", "bpm", "camelot", "energy"]
    file_exists = os.path.exists(LABELS_CSV)
    with open(LABELS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ── Retrain ───────────────────────────────────────────────────────────────────

def trigger_retrain(batch: int, epochs: int = 100):
    """Eğitimi arka planda başlatır (önceki eğitimi öldürür)."""
    venv_python = os.path.join(PROJECT_ROOT, "venv", "bin", "python")
    subprocess.run(["pkill", "-f", "src/train.py"], capture_output=True)
    time.sleep(2)

    log_path = f"/tmp/train_cycle{batch}.log"
    cmd = [
        venv_python, TRAIN_SCRIPT,
        "--epochs", str(epochs),
        "--batch", "32",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        cwd=PROJECT_ROOT,
    )
    print(f"\n  [RETRAIN] PID {proc.pid} → {log_path}")
    return proc


# ── Ana döngü ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SoundCloud Scraper + Eğitim Pipeline")
    parser.add_argument("--genres",     nargs="+", default=DEFAULT_GENRES)
    parser.add_argument("--per-genre",  type=int,  default=300,
                        help="Genre başına aranacak track sayısı")
    parser.add_argument("--retrain-every", type=int, default=RETRAIN_EVERY,
                        help="Kaç yeni örnekte bir yeniden eğit")
    parser.add_argument("--no-retrain", action="store_true",
                        help="Yalnızca veri topla, eğitme")
    args = parser.parse_args()

    print("=" * 60)
    print("SoundCloud Scraper + Otomatik Eğitim Pipeline")
    print("=" * 60)
    print(f"  Genre sayısı   : {len(args.genres)}")
    print(f"  Genre başına   : {args.per_genre} track aranacak")
    print(f"  Retrain her    : {args.retrain_every} yeni örnek")
    print(f"  Depolama mod   : MP3 indir → cache al → MP3 sil\n")

    existing = load_existing_labels()
    print(f"Mevcut etiket sayısı: {len(existing)}")

    new_count     = 0
    retrain_count = 0
    train_proc    = None
    skipped_no_key = 0

    # ── Sorgu stratejisi ──────────────────────────────────────────────────────
    # 1) Genre + Camelot kodu  ("afro house 8A")
    # 2) Genre + müzikal key   ("afro house A minor", "afro house F# major")
    # 3) Doğrudan key araması  ("A minor mix", "F# major house")
    # 4) BPM + key             ("128 bpm A minor", "124 bpm 8A")

    camelot_codes = list(CAMELOT_MAP.keys())

    # Müzikal key adları (kısa ve uzun biçim)
    key_names_short = [
        "Am", "Bm", "Cm", "Dm", "Em", "Fm", "Gm",
        "F#m", "C#m", "G#m", "D#m", "A#m",
        "A", "B", "C", "D", "E", "F", "G",
        "F#", "C#", "G#", "D#", "A#",
    ]
    key_names_long = [
        "A minor", "B minor", "C minor", "D minor", "E minor",
        "F minor", "G minor", "F# minor", "C# minor", "G# minor",
        "D# minor", "A# minor",
        "A major", "B major", "C major", "D major", "E major",
        "F major", "G major", "F# major", "C# major", "G# major",
        "D# major", "A# major",
    ]
    bpm_values = ["120", "124", "126", "128", "130", "132", "138", "140", "145", "150", "174"]

    queries = []

    # 1) Genre + Camelot
    for genre in args.genres:
        for code in camelot_codes:
            queries.append((genre, f"{genre} {code}"))

    # 2) Genre + müzikal key adı (uzun biçim — en güvenilir parse)
    for genre in args.genres:
        for key in key_names_long:
            queries.append((genre, f"{genre} {key}"))

    # 3) Doğrudan key araması (genre bağımsız)
    for key in key_names_long:
        queries.append(("key_search", f"{key} dj mix"))
        queries.append(("key_search", f"{key} house"))
        queries.append(("key_search", f"{key} techno"))

    # 4) BPM + Camelot (DJ set'lerde yaygın)
    for bpm in bpm_values:
        for code in camelot_codes[:12]:  # ilk 12 kod yeterli
            queries.append(("bpm_key", f"{bpm} bpm {code}"))

    random.shuffle(queries)
    print(f"Toplam arama sorgusu: {len(queries)}")

    for genre, query in queries:
        n_per_query = max(10, args.per_genre // (len(camelot_codes) + 1))
        print(f"\n{'─'*50}")
        print(f"Arama: '{query}' ({n_per_query} track)")
        tracks = scrape_soundcloud_query(query, n=n_per_query)
        print(f"  Bulunan track: {len(tracks)}")

        ok_genre = 0
        for track in tracks:
            sc_id = track["sc_id"] or re.sub(r"[^a-zA-Z0-9]", "_", track["title"])[:40]
            filename = f"sc_{sc_id}.mp3"
            cache_path = os.path.join(CACHE_DIR, f"sc_{sc_id}.npy")

            # Zaten cache'te varsa atla
            if filename in existing or os.path.exists(cache_path):
                continue

            # DJ track filtresi — okul sınıfı, oyun vb. içerikleri elek
            if not is_dj_track(track):
                skipped_no_key += 1
                continue

            # Key çıkar
            key = extract_key_from_metadata(track)
            if not key:
                skipped_no_key += 1
                continue

            # MP3 indir
            mp3_path = os.path.join(PREVIEW_DIR, filename)
            if not download_clip(track["url"], mp3_path):
                continue

            # Ses yükle + BPM + Chroma
            try:
                y, sr = librosa.load(mp3_path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
                bpm = extract_bpm(y, sr)
                chroma = extract_all_features(mp3_path, target_length=TARGET_LENGTH)
                np.save(cache_path, chroma)
            except Exception as e:
                if os.path.exists(mp3_path):
                    os.remove(mp3_path)
                continue

            # MP3 sil — sadece .npy cache kalır
            os.remove(mp3_path)

            # CSV'ye ekle
            row = {
                "song_id":     f"sc_{sc_id}",
                "filename":    filename,
                "musical_key": key,
                "bpm":         round(bpm, 1),
                "camelot":     "",
                "energy":      "",
            }
            append_to_csv([row])
            existing[filename] = row

            new_count += 1
            ok_genre  += 1
            print(f"  + {key:<14} {round(bpm):>3} BPM  {track['title'][:45]}")

            # Retrain tetikle
            if not args.no_retrain and new_count % args.retrain_every == 0:
                retrain_count += 1
                # Önceki eğitimi bekle
                if train_proc and train_proc.poll() is None:
                    print(f"\n  [RETRAIN] Önceki eğitim hâlâ devam ediyor, bekleniyor...")
                    train_proc.wait()
                train_proc = trigger_retrain(retrain_count)

            time.sleep(random.uniform(0.3, 0.8))

        print(f"  Genre sonucu: {ok_genre} yeni örnek eklendi")

    # Son özet
    print("\n" + "=" * 60)
    print("Pipeline tamamlandı!")
    print(f"  Yeni örnek      : {new_count}")
    print(f"  Key bulunamadı  : {skipped_no_key} (atlandı)")
    print(f"  Toplam dataset  : {len(existing)}")
    print(f"  Retrain sayısı  : {retrain_count}")

    if not args.no_retrain and new_count > 0:
        print(f"\nSon retrain başlatılıyor ({len(existing)} örnekle)...")
        if train_proc and train_proc.poll() is None:
            train_proc.wait()
        trigger_retrain(retrain_count + 1)

    print("=" * 60)


if __name__ == "__main__":
    main()
