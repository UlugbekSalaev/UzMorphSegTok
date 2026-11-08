"""Step 3. Train BPE and Unigram tokenizers (HuggingFace Tokenizers).

  bpe_raw_V, uni_raw_V: on the raw corpus;
  bpe_cse_V, uni_cse_V: on the corpus segmented by CSE-Seg+P (pieces do not cross morpheme boundaries);
  V = 8000, 16000, 32000, 48000.
Tokenizers on the corpus segmented by FEMSeg-uz are trained in step 8 (bpe_fem_32000, uni_fem_32000).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from uzseg.tokenizers import train_tokenizer


def main():
    for v in [C.VOCAB_SIZE] + [x for x in C.VOCAB_SWEEP if x != C.VOCAB_SIZE]:
        for mt, short in (('bpe', 'bpe'), ('unigram', 'uni')):
            for corpus, src in (('raw', 'train.txt'), ('cse', 'train_cse.txt')):
                out = os.path.join(C.TOK_DIR, f'{short}_{corpus}_{v}.json')
                if os.path.exists(out):
                    continue
                t = time.time()
                train_tokenizer(os.path.join(C.CORPUS_DIR, src), out, v, mt)
                print(f'{short}_{corpus}_{v}: {time.time() - t:.0f} s', flush=True)


if __name__ == '__main__':
    main()
