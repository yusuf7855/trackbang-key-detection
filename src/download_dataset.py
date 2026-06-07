"""
download_dataset.py
FMA (Free Music Archive) - Small dataset indirir.
8,000 şarkı, 8 tür, ~7GB (30sn clips)
https://github.com/mdeff/fma

Alternatif: GTZAN dataset (1,000 şarkı, 10 tür, ~1.2GB)

Kullanım: python download_dataset.py
"""

import os
import urllib.request
import zipfile

DATA_DIR = '../data/raw'

# FMA Small - 8k track, 30 saniyelik klipler
FMA_SMALL_URL = 'https://os.unil.cloud.switch.ch/fma/fma_small.zip'
FMA_METADATA_URL = 'https://os.unil.cloud.switch.ch/fma/fma_metadata.zip'

# GTZAN - alternatif, daha küçük
GTZAN_URL = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz'


def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = min(downloaded / total_size * 100, 100)
    mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    print(f'\r  {percent:.1f}% — {mb:.0f} / {total_mb:.0f} MB', end='', flush=True)


def download_fma_metadata():
    """FMA metadata'yı indir (track listesi, key, BPM bilgileri)"""
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, 'fma_metadata.zip')

    if os.path.exists(out_path):
        print('FMA metadata zaten indirilmiş.')
        return

    print('FMA metadata indiriliyor (~342MB)...')
    urllib.request.urlretrieve(FMA_METADATA_URL, out_path, show_progress)
    print('\nAçılıyor...')
    with zipfile.ZipFile(out_path, 'r') as z:
        z.extractall(DATA_DIR)
    print('FMA metadata hazır.')


def download_fma_small():
    """FMA Small ses dosyalarını indir"""
    out_path = os.path.join(DATA_DIR, 'fma_small.zip')

    if os.path.exists(os.path.join(DATA_DIR, 'fma_small')):
        print('FMA Small zaten mevcut.')
        return

    print('FMA Small indiriliyor (~7.2GB)...')
    print('Bu birkaç dakika sürebilir.')
    urllib.request.urlretrieve(FMA_SMALL_URL, out_path, show_progress)
    print('\nAçılıyor...')
    with zipfile.ZipFile(out_path, 'r') as z:
        z.extractall(DATA_DIR)
    print('FMA Small hazır.')


if __name__ == '__main__':
    print('=== FMA Dataset İndirici ===\n')
    print('NOT: fma_small ~7GB disk alanı gerektirir.')
    print('Önce sadece metadata indirilecek.\n')

    download_fma_metadata()

    ans = input('\nSes dosyalarını da indir? (~7GB) [e/h]: ').strip().lower()
    if ans == 'e':
        download_fma_small()
    else:
        print('\nSadece metadata indirildi.')
        print('Kendi mp3 dosyalarını data/raw/ klasörüne koyabilirsin.')
