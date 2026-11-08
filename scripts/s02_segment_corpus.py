"""Step 2. Segment the tokenizer corpus with CSE-Seg+P.

Results:
  data/corpus/word_freq.tsv        word types of the corpus and their frequencies;
  data/corpus/paradigm_stats.json  paradigm statistics of CSE-Seg+P (N(r), F(e)) and the selected weights;
  data/corpus/train_cse.txt        training corpus with morpheme boundaries marked by U+2AF6;
  data/corpus/silver_words.tsv     word types of the first 50,000 sentences with their segmentation (source C).
"""
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from uzseg.cse import read_rows, CSE, load_lexicon
from uzseg.rule_seg import Candidates, CSESeg, paradigm_stats
from uzseg.text import word_spans, normalize, is_word

DELIM = '⫶'          # morpheme boundary character (it does not occur in the corpus)


def load_segmenter():
    cse = CSE(read_rows(C.CSE_CSV))
    lex = load_lexicon(C.LEXICON)
    cands = Candidates(cse, lex, min_stem=2)
    stats = json.load(open(os.path.join(C.CORPUS_DIR, 'paradigm_stats.json'), encoding='utf-8'))
    prm = stats['params']
    return (CSESeg(cands, mode='paradigm', stats=stats, beta=prm['beta'], gamma=prm['gamma']),
            CSESeg(cands, mode='longest'), cse, lex)


def tune_params(cands, stats):
    """Select beta and gamma on the official train part of UzUDT: the highest boundary F1, then the highest exact match."""
    from uzseg.metrics import boundary_scores, boundaries
    dev = [(w['w'], w['seg']) for line in open(os.path.join(C.REF_DIR, C.DEV_SET + '.jsonl'), encoding='utf-8')
           for w in json.loads(line)['words']]
    grid = []
    for beta in (0.0, 0.1, 0.25, 0.5, 1.0):
        for gamma in (0.0, 1.0, 2.0, 3.0):
            seg = CSESeg(cands, mode='paradigm', stats=stats, beta=beta, gamma=gamma)
            sc = boundary_scores([(w, g, boundaries(seg.segment(w))) for w, g in dev])
            grid.append({'beta': beta, 'gamma': gamma, 'dev_f1': sc['f1'], 'dev_exact': sc['exact']})
    best = max(grid, key=lambda x: (round(x['dev_f1'], 6), round(x['dev_exact'], 6), -x['beta'], -x['gamma']))
    return best, grid


def segment_sentence(s, segment):
    """Insert the boundary character inside every word of a sentence (letter case is kept)."""
    out, last = [], 0
    for a, b in word_spans(s):
        out.append(s[last:a])
        word = s[a:b]
        w = normalize(word)
        if is_word(w) and len(w) == len(word):
            seg = segment(w)
            pos, cuts = 0, set()
            for m in seg[:-1]:
                pos += len(m)
                cuts.add(pos)
            word = ''.join((DELIM if i in cuts else '') + ch for i, ch in enumerate(word))
        out.append(word)
        last = b
    out.append(s[last:])
    return ''.join(out)


def main():
    t0 = time.time()
    train = open(os.path.join(C.CORPUS_DIR, 'train.txt'), encoding='utf-8').read().splitlines()
    freq = Counter()
    for s in train:
        for a, b in word_spans(s):
            w = normalize(s[a:b])
            if is_word(w):
                freq[w] += 1
    with open(os.path.join(C.CORPUS_DIR, 'word_freq.tsv'), 'w', encoding='utf-8') as f:
        for w, c in freq.most_common():
            f.write(f'{w}\t{c}\n')
    cse = CSE(read_rows(C.CSE_CSV))
    cands = Candidates(cse, load_lexicon(C.LEXICON), min_stem=2)
    stats = paradigm_stats(cands, freq)
    best, grid = tune_params(cands, stats)
    stats['params'], stats['grid'] = best, grid
    json.dump(stats, open(os.path.join(C.CORPUS_DIR, 'paradigm_stats.json'), 'w', encoding='utf-8'))
    print(f'word types: {len(freq)}, paradigm stems: {len(stats["stem_types"])}; selected: {best} '
          f'({time.time() - t0:.0f} s)', flush=True)

    seg = CSESeg(cands, mode='paradigm', stats=stats, beta=best['beta'], gamma=best['gamma'])
    with open(os.path.join(C.CORPUS_DIR, 'train_cse.txt'), 'w', encoding='utf-8') as f:
        for s in train:
            f.write(segment_sentence(s, seg.segment) + '\n')
    print(f'corpus segmented ({time.time() - t0:.0f} s)', flush=True)

    # corpus words of FEMSeg-uz: word types of the first 50,000 sentences
    sub = Counter()
    for s in train[:C.SEG_TRAIN_SENTENCES]:
        for a, b in word_spans(s):
            w = normalize(s[a:b])
            if is_word(w) and 2 <= len(w) <= 30:
                sub[w] += 1
    with open(os.path.join(C.CORPUS_DIR, 'silver_words.tsv'), 'w', encoding='utf-8') as f:
        for w, c in sub.most_common():
            f.write(f"{w}\t{c}\t{'|'.join(seg.segment(w))}\n")
    tokens = sum(sub.values())
    print(f'corpus words: {len(sub)} types, {tokens} tokens ({time.time() - t0:.0f} s)', flush=True)
    json.dump({'word_types': len(freq), 'silver_types': len(sub), 'silver_tokens': tokens,
               'silver_sentences': C.SEG_TRAIN_SENTENCES},
              open(os.path.join(C.RESULTS, 'silver_stats.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
