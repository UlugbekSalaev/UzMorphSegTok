"""Repeat the whole experiment from the beginning.

    python run_all.py            all steps (finished results are not computed again)
    python run_all.py 5 9        only the steps 5 to 9

Steps:
  0  download UD_Uzbek-UzUDT r2.18, the Uzbek News Dataset v0.0.1 and the apertium-uzb lexicon, check the checksums
  1  prepare the data (CSE table and its morphemes, reference set, tokenizer corpus, classification data)
  2  segment the corpus with CSE-Seg+P (beta and gamma are selected on the UzUDT train part)
  3  BPE and Unigram tokenizers (raw and CSE-segmented corpus, V = 8K, 16K, 32K, 48K)
  4  FEMSeg-uz: 8 configurations x 3 seeds and five-fold cross-validation of the main model (2 parallel processes)
  5  evaluate the segmenters (Morfessor, Morfessor+UD, UzMorphAnalyser, CSE-Seg, CSE-Seg+P, CSE-Seg+P+UD)
  6  tokenizers on the corpus segmented by FEMSeg-uz
  7  evaluate the tokenizers (raw and segmented modes) and the table of examples
  8  news classification (11 tokenizers x 3 seeds, check with 5,000 training documents per category) and the
     check of the saved models
  9  statistical analysis (confidence intervals, McNemar tests, stem boundary, seen and unseen words)
 10  data figures of the paper (paper/figures.py)
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = {
    0: [['scripts/s00_download_data.py']],
    1: [['scripts/s01_prepare_data.py']],
    2: [['scripts/s02_segment_corpus.py']],
    3: [['scripts/s03_train_tokenizers.py']],
    4: [['scripts/run_femseg_all.py', '2', '2']],
    5: [['scripts/s06_eval_segmenters.py']],
    6: [['scripts/s08_femseg_tokenizers.py', 'full_s17']],
    7: [['scripts/s05_eval_tokenizers.py'], ['scripts/s09_examples.py']],
    8: [['scripts/s07_extrinsic.py'], ['scripts/s07_extrinsic.py', '--large', 'bpe_raw_32000', 'bpe_cse_32000+seg'],
        ['scripts/check_models.py']],
    9: [['scripts/s10_analysis.py']],
    10: [['paper/figures.py']],
}


def run(cmd):
    print('>>>', ' '.join(cmd), flush=True)
    subprocess.run([sys.executable] + cmd, cwd=ROOT, check=True)


def main():
    first, last = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 10)
    for step in range(first, last + 1):
        for cmd in STEPS[step]:
            run(cmd)


if __name__ == '__main__':
    main()
