"""Step 7a. Evaluate the tokenizers.

  * held-out corpus (5,000 sentences): observed vocabulary, number of tokens, compression, bits per token
    (theoretical, empirical), information volume, sentence length in tokens (mean, median, maximum), tokens per word;
  * reference set: agreement of token boundaries inside words with the morpheme boundaries (P, R, F1, exact match),
    the same scores with the boundaries between words counted, the stem boundary and the share of words kept whole.
Result: results/tokenizers_intrinsic.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from uzseg.metrics import boundary_scores, boundaries, information_metrics, length_stats
from uzseg.text import word_spans
from uzseg.tokenizers import load_tokenizers, word_cuts


def ud_sentences(name):
    """Reference sentences: text (UD tokens joined by spaces), word positions and reference morphemes."""
    out = []
    for line in open(os.path.join(C.REF_DIR, name + '.jsonl'), encoding='utf-8'):
        s = json.loads(line)
        text, spans = '', []
        for f in s['forms']:
            if text:
                text += ' '
            spans.append((len(text), len(text) + len(f)))
            text += f
        words = [(w['w'], w['seg'], spans[w['i']]) for w in s['words'] if len(w['form']) == len(w['w'])]
        out.append((text, words))
    return out


def boundary_eval(tok, sentences):
    pairs, per_sentence = [], []
    for text, words in sentences:
        enc = tok.encode(text)
        cuts = word_cuts(enc, [sp for _, _, sp in words])
        sp = [(w, seg, c) for (w, seg, _), c in zip(words, cuts)]
        pairs += sp
        sc = boundary_scores(sp) if sp else {'TP': 0, 'FP': 0, 'FN': 0, 'exact': 0.0, 'words': 0}
        per_sentence.append([sc['TP'], sc['FP'], sc['FN'], sc['words'], round(sc['exact'] * sc['words'])])
    r = boundary_scores(pairs)
    # stem boundary: is there a token boundary between the stem and the ending of a segmented reference word?
    stem = [len(seg[0]) in c for w, seg, c in pairs if len(seg) > 1]
    r['stem_recall'] = sum(stem) / len(stem)
    # segmented reference words that the tokenizer keeps as one token
    r['kept_whole'] = sum(1 for w, seg, c in pairs if len(seg) > 1 and not c) / len(stem)
    # share of extra (FP) boundaries that fall inside the stem (the rest fall inside the suffix chain)
    fp = [(p < len(seg[0])) for w, seg, c in pairs for p in c - boundaries(seg)]
    r['fp_in_stem'] = sum(fp) / len(fp) if fp else 0.0
    # segmented mode: share of the extra boundaries that come from the segmenter itself
    segmenter = getattr(tok, 'segmenter', None)
    if segmenter is not None:
        inh = [p in segmenter(w) for w, seg, c in pairs for p in c - boundaries(seg)]
        r['fp_from_segmenter'] = sum(inh) / len(inh) if inh else 0.0
    r['per_sentence'] = per_sentence
    return r


def intrinsic(tok, sentences):
    token_lists, lengths, n_words, n_chars, split_words, word_tokens = [], [], 0, 0, 0, 0
    for s in sentences:
        enc = tok.encode(s)
        token_lists.append([t for t, _, _ in enc])
        lengths.append(len(enc))
        n_chars += len(s)
        spans = word_spans(s)
        cuts = word_cuts(enc, spans)
        n_words += len(spans)
        split_words += sum(1 for c in cuts if c)
        word_tokens += sum(1 + len(c) for c in cuts)
    m = information_metrics(token_lists, n_chars)
    m['tokens_per_word'] = word_tokens / n_words          # tokens that cover one word (word-internal pieces)
    m['tokens_per_word_all'] = sum(lengths) / n_words      # all tokens of the text divided by the number of words
    m['split_word_share'] = split_words / n_words
    m['length'] = length_stats(lengths)
    m['lengths'] = lengths
    return m


def all_names():
    """All tokenizers: our models (those of segmented corpora in both modes) and the multilingual ones."""
    names = []
    for f in sorted(os.listdir(C.TOK_DIR)):
        if f.endswith('.json'):
            n = f[:-5]
            names.append(n)
            if '_cse_' in n or '_fem_' in n:
                names.append(n + '+seg')
    return names + ['mT5', 'XLM-R', 'mBERT']


def main(names=None):
    test = open(os.path.join(C.CORPUS_DIR, 'test.txt'), encoding='utf-8').read().splitlines()
    ref = {ts: ud_sentences(ts) for ts in C.TEST_SETS}
    out_path = os.path.join(C.RESULTS, 'tokenizers_intrinsic.json')
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    todo = names or [n for n in all_names() if n not in results]
    for name, tok in load_tokenizers(C.TOK_DIR, todo).items():
        r = {'vocab_size': tok.vocab_size, 'intrinsic': intrinsic(tok, test),
             'boundary': {ts: boundary_eval(tok, ref[ts]) for ts in C.TEST_SETS}}
        results[name] = r
        b = r['boundary'][C.TEST_SETS[0]]
        print(f"{name:20s} tokens {r['intrinsic']['tokens']:7d}  t/w {r['intrinsic']['tokens_per_word']:.3f}  "
              f"P {b['precision']:.3f} R {b['recall']:.3f} F1 {b['f1']:.3f} text F1 {b['text_f1']:.3f}", flush=True)
        json.dump(results, open(out_path, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:] or None)
