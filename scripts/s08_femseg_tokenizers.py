"""Step 6. BPE and Unigram tokenizers on the corpus segmented by FEMSeg-uz (bpe_fem_32000, uni_fem_32000).

The main FEMSeg-uz model (full, seed 17, CSE-constrained decoding) segments every word type of the corpus, and the
tokenizers are then trained as in step 3.
Usage: python scripts/s08_femseg_tokenizers.py [model name]
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from scripts.s02_segment_corpus import segment_sentence
from uzseg.femseg import load_model, predict_cuts
from uzseg.segmenters import candidates
from uzseg.text import word_spans, normalize, is_word
from uzseg.tokenizers import train_tokenizer


def load_femseg(name):
    """(model, character vocabulary, checkpoint) for other scripts."""
    path = os.path.join(C.SEG_DIR, name + '.pt')
    model, vocab = load_model(path)
    return model, vocab, torch.load(path, weights_only=False)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'full_s17'
    torch.set_num_threads(4)
    t0 = time.time()
    model, vocab = load_model(os.path.join(C.SEG_DIR, name + '.pt'))
    cands = candidates()
    train = open(os.path.join(C.CORPUS_DIR, 'train.txt'), encoding='utf-8').read().splitlines()
    types = sorted({normalize(s[a:b]) for s in train for a, b in word_spans(s)})
    types = [w for w in types if is_word(w) and len(w) <= 40]
    cuts = predict_cuts(model, types, vocab, cands, 'hard', bs=1024)
    segs = {}
    for w, c in zip(types, cuts):
        bounds = sorted(c) + [len(w)]
        last, seg = 0, []
        for b in bounds:
            seg.append(w[last:b])
            last = b
        segs[w] = tuple(seg)
    print(f'{len(types)} word types segmented ({time.time() - t0:.0f} s)', flush=True)
    path = os.path.join(C.CORPUS_DIR, 'train_fem.txt')
    with open(path, 'w', encoding='utf-8') as f:
        for s in train:
            f.write(segment_sentence(s, lambda w: segs.get(w, (w,))) + '\n')
    for mt, short in (('bpe', 'bpe'), ('unigram', 'uni')):
        out = os.path.join(C.TOK_DIR, f'{short}_fem_{C.VOCAB_SIZE}.json')
        if not os.path.exists(out):
            train_tokenizer(path, out, C.VOCAB_SIZE, mt)
            print(f'{out}: done ({time.time() - t0:.0f} s)', flush=True)


if __name__ == '__main__':
    main()
