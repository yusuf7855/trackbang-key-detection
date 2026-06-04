"""
build_dataset.py
================
TrackBang MongoDB'sinden şarkı verilerini çeker, SoundCloud'dan ses indirir,
35 saniyelik klip keser ve eğitim için labels.csv oluşturur.

Akış:
  1. MongoDB → SoundCloud URL + musicalKey/bpm/camelot
  2. yt-dlp ile SoundCloud'dan tam şarkı indir (tmp)
  3. ffmpeg ile 30s–65s arasını kes → data/raw/previews/<id>.mp3
  4. data/labels.csv kaydet

Kullanım:
  ssh -i ~/.ssh/id_ed25519 -L 27017:127.0.0.1:27017 root@72.62.63.184 -N -f
  python src/build_dataset.py
  python src/build_dataset.py --limit 100   # Test için ilk 100 şarkı
"""

import os
import sys
import csv
import re
import time
import random
import subprocess
import tempfile
import argparse
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pymongo import MongoClient

# ── Yapılandırma ──────────────────────────────────────────────────────────────

MONGO_URI   = "mongodb://127.0.0.1:27017/trackbang"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREVIEW_DIR  = os.path.join(PROJECT_ROOT, "data", "raw", "previews")
LABELS_CSV   = os.path.join(PROJECT_ROOT, "data", "labels.csv")

os.makedirs(PREVIEW_DIR, exist_ok=True)

# Klibin başlangıç noktası
# Not: SoundCloud artık çoğu şarkı için 30sn preview döndürüyor.
# Preview dosyaları için CLIP_OFFSET=0 gerekir; tam track için 30 kullanılabilir.
CLIP_OFFSET   = 0    # sn — preview uyumluluğu için baştan al
CLIP_DURATION = 30   # sn — kaç saniye al

YTDLP  = os.path.join(PROJECT_ROOT, "venv", "bin", "yt-dlp")
FFMPEG = "ffmpeg"

# ── Key normalleştirme ────────────────────────────────────────────────────────

VALID_KEYS = {
    'C Major', 'C Minor', 'C# Major', 'C# Minor',
    'D Major', 'D Minor', 'D# Major', 'D# Minor',
    'E Major', 'E Minor', 'F Major', 'F Minor',
    'F# Major', 'F# Minor', 'G Major', 'G Minor',
    'G# Major', 'G# Minor', 'A Major', 'A Minor',
    'A# Major', 'A# Minor', 'B Major', 'B Minor',
}

ENHARMONIC = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}

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


def normalize_key(raw_key: Optional[str], camelot: Optional[str]) -> Optional[str]:
    if raw_key:
        k = raw_key.strip()
        if k in VALID_KEYS:
            return k
        # "Am" / "F#m"
        if k.endswith('m') and not k.endswith('Major') and not k.endswith('Minor'):
            note = ENHARMONIC.get(k[:-1], k[:-1])
            c = f"{note} Minor"
            if c in VALID_KEYS:
                return c
        # "A" / "Db"
        if not any(w in k for w in ('Major', 'Minor', 'maj', 'min')):
            note = ENHARMONIC.get(k, k)
            c = f"{note} Major"
            if c in VALID_KEYS:
                return c
        # Büyük/küçük harf farklılığı
        for vk in VALID_KEYS:
            if k.lower() == vk.lower():
                return vk
        # "A min" / "F# maj"
        lo = k.lower()
        for suffix, mode in ((' min', 'Minor'), (' maj', 'Major')):
            if lo.endswith(suffix):
                note = ENHARMONIC.get(k[:-len(suffix)].strip(), k[:-len(suffix)].strip())
                c = f"{note} {mode}"
                if c in VALID_KEYS:
                    return c

    if camelot:
        result = CAMELOT_MAP.get(str(camelot).strip().upper())
        if result:
            return result

    return None


# ── MongoDB ───────────────────────────────────────────────────────────────────

def fetch_songs_from_db(limit: Optional[int] = None) -> List[dict]:
    print(f"MongoDB bağlantısı: {MONGO_URI}")
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
        client.server_info()
    except Exception as e:
        print(f"\n[HATA] MongoDB bağlantısı kurulamadı: {e}")
        print("SSH tüneli açık mı?")
        print("  ssh -i ~/.ssh/id_ed25519 -L 27017:127.0.0.1:27017 root@72.62.63.184 -N -f")
        sys.exit(1)

    col = client["trackbang"]["musics"]

    query = {
        "platformLinks.soundcloud": {"$exists": True, "$ne": None, "$ne": ""},
        "$or": [
            {"musicalKey": {"$exists": True, "$ne": None}},
            {"camelot":    {"$exists": True, "$ne": None}},
        ]
    }
    cursor = col.find(query, {
        "platformLinks.soundcloud": 1,
        "musicalKey": 1, "bpm": 1, "camelot": 1, "energy": 1,
        "title": 1, "artist": 1, "_id": 1,
    })
    if limit:
        cursor = cursor.limit(limit)

    songs = []
    for doc in cursor:
        sc_url = (doc.get("platformLinks") or {}).get("soundcloud")
        if not sc_url:
            continue
        # Geçersiz URL'leri filtrele (artist sayfası, search URL'leri)
        clean = re.sub(r'[?#].*', '', sc_url.rstrip('/'))
        if len(clean.split('/')) < 5:  # soundcloud.com/artist/track → 5 parça
            continue
        norm_key = normalize_key(doc.get("musicalKey"), doc.get("camelot"))
        if not norm_key:
            continue
        songs.append({
            "id":         str(doc["_id"]),
            "sc_url":     sc_url,
            "key":        norm_key,
            "bpm":        float(doc["bpm"]) if doc.get("bpm") else None,
            "camelot":    doc.get("camelot"),
            "energy":     float(doc["energy"]) if doc.get("energy") else None,
        })

    client.close()
    print(f"{len(songs)} şarkı bulundu (SoundCloud URL + geçerli key).")
    return songs


# ── İndirme ve kırpma ─────────────────────────────────────────────────────────

def download_and_clip(sc_url: str, dest_path: str, retries: int = 2) -> bool:
    """
    SoundCloud URL'sini yt-dlp ile indirir,
    ffmpeg ile CLIP_OFFSET–(CLIP_OFFSET+CLIP_DURATION) aralığını keser,
    dest_path'e MP3 olarak kaydeder.
    """
    for attempt in range(retries + 1):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_tmpl = os.path.join(tmpdir, "audio.%(ext)s")

            dl_cmd = [
                YTDLP,
                sc_url,
                "-x", "--audio-format", "mp3", "--audio-quality", "5",
                "-o", tmp_tmpl,
                "--quiet", "--no-warnings",
                "--no-playlist",
            ]
            try:
                result = subprocess.run(
                    dl_cmd, capture_output=True, text=True, timeout=45
                )
                if result.returncode != 0:
                    if attempt < retries:
                        time.sleep(random.uniform(2, 5))
                        continue
                    return False
            except subprocess.TimeoutExpired:
                if attempt < retries:
                    time.sleep(random.uniform(2, 4))
                    continue
                return False

            tmp_files = [f for f in os.listdir(tmpdir) if f.startswith("audio.")]
            if not tmp_files:
                if attempt < retries:
                    continue
                return False
            tmp_audio = os.path.join(tmpdir, tmp_files[0])

            clip_cmd = [
                FFMPEG, "-y",
                "-i", tmp_audio,
                "-ss", str(CLIP_OFFSET),
                "-t",  str(CLIP_DURATION),
                "-acodec", "libmp3lame", "-q:a", "5",
                dest_path,
                "-loglevel", "error",
            ]
            try:
                result = subprocess.run(
                    clip_cmd, capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0 and os.path.getsize(dest_path) > 10_000:
                    return True
            except subprocess.TimeoutExpired:
                pass

            if attempt < retries:
                time.sleep(random.uniform(1, 3))

    return False


def _worker(song: dict) -> tuple:
    """Thread worker — (id, ok) döndürür."""
    dest = os.path.join(PREVIEW_DIR, f"{song['id']}.mp3")
    # Worker'lar arası çakışmayı azaltmak için küçük rastgele bekleme
    time.sleep(random.uniform(0, 1.5))
    ok = download_and_clip(song["sc_url"], dest)
    return song["id"], ok


# ── Ana akış ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Test için kaç şarkı indir (default: hepsi)")
    parser.add_argument("--workers", type=int, default=3,
                        help="Paralel indirme sayısı (default: 3)")
    args = parser.parse_args()

    print("=" * 60)
    print("TrackBang Dataset Builder — SoundCloud")
    print("=" * 60)
    if args.limit:
        print(f"[TEST MODU] Limit: {args.limit} şarkı")

    # 1) DB'den şarkıları çek
    songs = fetch_songs_from_db(limit=args.limit)
    if not songs:
        print("[HATA] Hiç şarkı bulunamadı.")
        sys.exit(1)

    # 2) Zaten indirilmiş veya cache'i olanları filtrele
    cache_dir = os.path.join(PROJECT_ROOT, "data", "processed", "cache")
    pending = []
    already  = 0
    for s in songs:
        dest       = os.path.join(PREVIEW_DIR, f"{s['id']}.mp3")
        cache_path = os.path.join(cache_dir, f"{s['id']}.npy")
        # MP3 varsa veya .npy cache varsa atla
        if (os.path.exists(dest) and os.path.getsize(dest) > 10_000) \
                or os.path.exists(cache_path):
            already += 1
        else:
            pending.append(s)

    print(f"Zaten mevcut (MP3/cache): {already} | İndirilecek: {len(pending)}")

    # 3) İndir
    failed  = []
    success = 0

    if pending:
        print(f"\nKlipler indiriliyor → {PREVIEW_DIR}")
        print(f"Her klip: {CLIP_OFFSET}s–{CLIP_OFFSET+CLIP_DURATION}s ({CLIP_DURATION}sn)")
        print(f"Paralel worker: {args.workers}\n")

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, s): s for s in pending}
            with tqdm(total=len(pending), desc="İndiriliyor", unit="şarkı") as bar:
                for fut in as_completed(futures):
                    sid, ok = fut.result()
                    if ok:
                        success += 1
                    else:
                        failed.append(sid)
                        tqdm.write(f"  [!] Başarısız: {sid}")
                    bar.update(1)
                    bar.set_postfix(ok=success, fail=len(failed))

        print(f"\nBaşarılı: {success} | Başarısız: {len(failed)}")

    # 4) CSV yaz (indirilen + önceden mevcut olan)
    rows = []
    for s in songs:
        dest = os.path.join(PREVIEW_DIR, f"{s['id']}.mp3")
        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            rows.append({
                "song_id":     s["id"],
                "filename":    f"{s['id']}.mp3",
                "musical_key": s["key"],
                "bpm":         s["bpm"] if s["bpm"] else "",
                "camelot":     s["camelot"] if s["camelot"] else "",
                "energy":      s["energy"] if s["energy"] else "",
            })

    fieldnames = ["song_id", "filename", "musical_key", "bpm", "camelot", "energy"]
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 5) Özet
    from collections import Counter
    dist = Counter(r["musical_key"] for r in rows)

    print("\n" + "=" * 60)
    print(f"Dataset tamamlandı!")
    print(f"  MP3 klasörü  : {PREVIEW_DIR}")
    print(f"  Etiket CSV   : {LABELS_CSV}")
    print(f"  Toplam örnek : {len(rows)}")
    print(f"\nKey dağılımı (ilk 10):")
    for key, cnt in dist.most_common(10):
        bar = "█" * (cnt * 25 // max(dist.values()))
        print(f"  {key:<14} {bar} {cnt}")
    if failed:
        print(f"\n[!] {len(failed)} şarkı indirilemedi — bir sonraki çalıştırmada tekrar denenir.")
    print("\nSıradaki adım:")
    print("  python src/train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
