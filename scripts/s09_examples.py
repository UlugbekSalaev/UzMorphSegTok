"""Step 7b. Tokenization examples of the paper: how selected words are divided by every main tokenizer.

Result: results/tokenization_examples.json  {word: {'CSE-Seg+P': 'kitob|lar|imiz|da', tokenizer: 'kitob|lari|mizda'}}
Usage: python scripts/s09_examples.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from scripts.s02_segment_corpus import load_segmenter
from uzseg.tokenizers import load_tokenizers, word_cuts

WORDS = ['kitoblarimizda', 'yuragimdagi', "o'qituvchilarimizning", 'ishlayotganlarning']
TOKENIZERS = ['bpe_raw_32000', 'uni_raw_32000', 'bpe_cse_32000', 'uni_cse_32000', 'bpe_cse_32000+seg',
              'uni_cse_32000+seg', 'bpe_fem_32000+seg', 'uni_fem_32000+seg', 'mT5', 'XLM-R', 'mBERT']


def mark(w, cuts):
    out = []
    for i, ch in enumerate(w):
        if i in cuts:
            out.append('|')
        out.append(ch)
    return ''.join(out)


def main():
    seg_p = load_segmenter()[0]
    out = {w: {'CSE-Seg+P': '|'.join(seg_p.segment(w))} for w in WORDS}
    names = [n for n in TOKENIZERS if n in ('mT5', 'XLM-R', 'mBERT') or
             os.path.exists(os.path.join(C.TOK_DIR, n.replace('+seg', '') + '.json'))]
    for name, tok in load_tokenizers(C.TOK_DIR, names).items():
        for w in WORDS:
            cuts = word_cuts(tok.encode(w), [(0, len(w))])[0]
            out[w][name] = mark(w, cuts)
    json.dump(out, open(os.path.join(C.RESULTS, 'tokenization_examples.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    for w, d in out.items():
        print(w, d)


if __name__ == '__main__':
    main()
