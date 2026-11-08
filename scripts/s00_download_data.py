"""Step 0. Download the external data and verify the checksums.

  * UD_Uzbek-UzUDT, release r2.18 (manual lemmas and features; CC BY-SA 4.0);
  * Uzbek News Dataset v0.0.1, Zenodo DOI 10.5281/zenodo.7677431 (raw text and topic labels; CC BY 4.0);
  * the stem lexicon of apertium-uzb at a fixed commit (GPL-3.0).
The UzMorphAnalyser files are part of the repository (data/cse/uzmorphanalyser_0.0.22) and are checked in step 1.
"""
import hashlib
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def digest(path, algo='sha256'):
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fetch(url, path):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    tmp = path + '.part'
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, 'wb') as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
    os.replace(tmp, path)
    print('downloaded', path, os.path.getsize(path), flush=True)


def main():
    ok = True
    for name, url in C.UD_URLS.items():
        path = os.path.join(C.UD_DIR, name)
        fetch(url, path)
        h = digest(path)
        expected = C.UD_SHA256.get(name)
        good = expected is None or h == expected
        ok &= good
        print(f'{name}: sha256 {h} {"OK" if good else "MISMATCH"}', flush=True)
    for name, expected in C.APERTIUM_FILES.items():
        path = os.path.join(C.APERTIUM_DIR, name)
        fetch(C.APERTIUM_BASE + name, path)
        h = digest(path)
        ok &= h == expected
        print(f'{name}: sha256 {h} {"OK" if h == expected else "MISMATCH"}', flush=True)
    fetch(C.NEWS_URL, C.NEWS_ZIP)
    h = digest(C.NEWS_ZIP, 'md5')
    good = h == C.NEWS_MD5
    ok &= good
    print(f'{os.path.basename(C.NEWS_ZIP)}: md5 {h} {"OK" if good else "MISMATCH"}', flush=True)
    if not ok:
        sys.exit('checksum mismatch')


if __name__ == '__main__':
    main()
