"""Segmenters used while a text is encoded.

A segmenter takes a word (lower case, one apostrophe form) and returns its set of morpheme boundaries:
    'kitoblarimizda' -> {5, 8, 12}.
  CSESegmenter: CSE-Seg+P (CSE table, lexicon, paradigm statistics);
  FEMSegmenter: a saved FEMSeg-uz model with CSE-constrained decoding.
"""
import json
import os

import config as C
from .cse import read_rows, CSE, load_lexicon
from .metrics import boundaries
from .rule_seg import Candidates, CSESeg


def candidates():
    return Candidates(CSE(read_rows(C.CSE_CSV)), load_lexicon(C.LEXICON), min_stem=2)


class CSESegmenter:
    def __init__(self, cands=None):
        self.cands = cands or candidates()
        stats = json.load(open(os.path.join(C.CORPUS_DIR, 'paradigm_stats.json'), encoding='utf-8'))
        prm = stats['params']
        self.seg = CSESeg(self.cands, mode='paradigm', stats=stats, beta=prm['beta'], gamma=prm['gamma'])

    def prefetch(self, words):
        pass

    def __call__(self, w):
        return boundaries(self.seg.segment(w))


class FEMSegmenter:
    """FEMSeg-uz predictions; prefetch() computes many words in batches."""

    def __init__(self, name='full_s17', cands=None, threads=4):
        import torch
        from .femseg import load_model
        torch.set_num_threads(threads)
        self.cands = cands or candidates()
        self.model, self.vocab = load_model(os.path.join(C.SEG_DIR, name + '.pt'))
        self.cache = {}

    def prefetch(self, words):
        from .femseg import predict_cuts
        new = sorted({w for w in words if w not in self.cache})
        if new:
            for w, c in zip(new, predict_cuts(self.model, new, self.vocab, self.cands, 'hard', bs=1024)):
                self.cache[w] = set(c)

    def __call__(self, w):
        if w not in self.cache:
            self.prefetch([w])
        return self.cache[w]
