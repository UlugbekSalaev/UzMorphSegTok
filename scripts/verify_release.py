"""Check that a copy of the repository reproduces the data and the reported numbers.

    python scripts/verify_release.py           check (after steps 0-2: python run_all.py 0 2)
    python scripts/verify_release.py --write   write data/checksums.json (done once by the author)

The check compares
  1) the SHA-256 checksums of the prepared data files (corpus, classification data, reference set, lexicon) with
     data/checksums.json;
  2) the scores of the released models, computed again from the released files, with the result files:
     FEMSeg-uz (all saved models), CSE-Seg and CSE-Seg+P on the reference set, and the boundary scores of the main
     tokenizers.
"""
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

DATA_FILES = ['data/cse/stems.csv', 'data/cse/cse_morphs.tsv', 'data/reference/UzUDT_train.jsonl',
              'data/reference/UzUDT_test.jsonl', 'data/corpus/train.txt', 'data/corpus/test.txt',
              'data/corpus/train_cse.txt', 'data/corpus/silver_words.tsv', 'data/extrinsic/uztc_train.jsonl',
              'data/extrinsic/uztc_reserve.jsonl', 'data/extrinsic/uztc_dev.jsonl', 'data/extrinsic/uztc_test.jsonl',
              'data/splits/news_tokenizer_train.txt', 'data/splits/news_tokenizer_test.txt',
              'data/splits/news_classification_train.txt', 'data/splits/news_classification_reserve.txt',
              'data/splits/news_classification_dev.txt', 'data/splits/news_classification_test.txt']
CHECKSUMS = os.path.join(C.DATA, 'checksums.json')


def sha256(path):
    """SHA-256 of a text file in which a Windows line end (CR LF) counts as LF, so that the data prepared on any
    operating system give the same checksum."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for line in f:
            h.update(line[:-2] + b'\n' if line.endswith(b'\r\n') else line)
    return h.hexdigest()


def check_data():
    expected = json.load(open(CHECKSUMS))
    ok = True
    for rel in DATA_FILES:
        path = os.path.join(C.ROOT, rel)
        if not os.path.exists(path):
            print(f'  missing   {rel}')
            ok = False
            continue
        same = sha256(path) == expected[rel]
        ok &= same
        print(f"  {'same     ' if same else 'DIFFERENT'} {rel}")
    return ok


def check_scores():
    from scripts.s02_segment_corpus import load_segmenter
    from scripts.s04_train_femseg import load_ref, ref_types
    from scripts.s05_eval_tokenizers import boundary_eval, ud_sentences
    from uzseg.metrics import boundaries, boundary_scores
    from uzseg.tokenizers import load_tokenizers
    ok = True
    toks = ref_types(load_ref(C.TEST_SETS[0]))[1]
    seg_p, seg_l = load_segmenter()[:2]
    saved = json.load(open(os.path.join(C.RESULTS, 'segmenters.json')))
    for name, seg in (('CSE-Seg', seg_l), ('CSE-Seg+P', seg_p)):
        f1 = boundary_scores([(w, s, boundaries(seg.segment(w))) for w, s in toks])['f1']
        same = abs(f1 - saved[name][C.TEST_SETS[0]]['f1']) < 1e-9
        ok &= same
        print(f"  {'same     ' if same else 'DIFFERENT'} {name}: F1 {100 * f1:.2f}")
    saved_tok = json.load(open(os.path.join(C.RESULTS, 'tokenizers_intrinsic.json')))
    names = ['bpe_raw_32000', 'uni_raw_32000', 'bpe_cse_32000', 'uni_cse_32000', 'bpe_cse_32000+seg',
             'uni_cse_32000+seg', 'bpe_fem_32000+seg', 'uni_fem_32000+seg']
    ref = ud_sentences(C.TEST_SETS[0])
    for name, tok in load_tokenizers(C.TOK_DIR, names).items():
        r = boundary_eval(tok, ref)
        same = abs(r['f1'] - saved_tok[name]['boundary'][C.TEST_SETS[0]]['f1']) < 1e-9
        ok &= same
        print(f"  {'same     ' if same else 'DIFFERENT'} {name}: F1 {100 * r['f1']:.2f}, "
              f"whole-text F1 {100 * r['text_f1']:.2f}")
    import subprocess
    rc = subprocess.run([sys.executable, os.path.join(C.ROOT, 'scripts', 'check_models.py')], cwd=C.ROOT).returncode
    report = json.load(open(os.path.join(C.RESULTS, 'check_models.json')))
    ok &= rc == 0 and all(v['ok'] for v in report.values())
    return ok


def main():
    if '--write' in sys.argv:
        json.dump({rel: sha256(os.path.join(C.ROOT, rel)) for rel in DATA_FILES}, open(CHECKSUMS, 'w'), indent=1)
        print('written', CHECKSUMS)
        return
    print('1) prepared data')
    ok_data = check_data()
    print('2) scores of the released models')
    ok_scores = check_scores()
    print('RESULT:', 'everything reproduced' if ok_data and ok_scores else 'differences found (see above)')
    sys.exit(0 if ok_data and ok_scores else 1)


if __name__ == '__main__':
    main()
