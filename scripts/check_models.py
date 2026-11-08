"""Check the saved FEMSeg-uz models: every .pt file is loaded again, the test words are segmented again and the
scores are compared with the numbers in results/femseg/*.json.

Usage:  python scripts/check_models.py            (all models)
        python scripts/check_models.py full_s17   (one model)
"""
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from scripts.s04_train_femseg import ref_types, load_ref
from scripts.s08_femseg_tokenizers import load_femseg
from uzseg.cse import read_rows, CSE, load_lexicon
from uzseg.femseg import predict_cuts
from uzseg.metrics import boundary_scores
from uzseg.rule_seg import Candidates


def main():
    torch.set_num_threads(4)
    names = sys.argv[1:] or sorted(os.path.basename(p)[:-3] for p in glob.glob(os.path.join(C.SEG_DIR, '*.pt')))
    cands = Candidates(CSE(read_rows(C.CSE_CSV)), load_lexicon(C.LEXICON), min_stem=2)
    tests = {ts: ref_types(load_ref(ts))[1] for ts in C.TEST_SETS}
    words = sorted({w for toks in tests.values() for w, _ in toks})
    report, ok_all = {}, True
    for name in names:
        saved = json.load(open(os.path.join(C.RESULTS, 'femseg', name + '.json')))
        model, vocab, _ = load_femseg(name)
        pred = dict(zip(words, predict_cuts(model, words, vocab, cands, 'hard')))
        row = {}
        for ts, toks in tests.items():
            f1 = boundary_scores([(w, seg, pred[w]) for w, seg in toks])['f1']
            row[ts] = (round(f1, 6), round(saved['scores']['hard'][ts]['f1'], 6))
        ok = all(abs(a - b) < 1e-9 for a, b in row.values())
        ok_all &= ok
        report[name] = {'ok': ok, 'f1_recomputed_vs_saved': row}
        print(f"{name:18s} {'SAME' if ok else 'DIFFERENT'}  " +
              '  '.join(f"{ts} {100 * a:.2f}/{100 * b:.2f}" for ts, (a, b) in row.items()), flush=True)
    json.dump(report, open(os.path.join(C.RESULTS, 'check_models.json'), 'w'), indent=1)
    print('ALL THE SAME' if ok_all else 'WARNING: differences found')


if __name__ == '__main__':
    main()
