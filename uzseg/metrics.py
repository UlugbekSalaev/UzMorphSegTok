"""Evaluation measures: morpheme boundaries, edit distance and information measures of tokenizers."""
import math
from collections import Counter

import numpy as np


def boundaries(seg):
    """List of morphemes -> set of boundary positions: ['kitob','lar','da'] -> {5, 8}."""
    out, pos = set(), 0
    for m in seg[:-1]:
        pos += len(m)
        out.add(pos)
    return out


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def boundary_scores(pairs):
    """pairs: [(word, reference morphemes, predicted boundary set)].
    Each of the |w|-1 positions inside a word is one of TP, FP, FN or TN.
    The whole-text scores also count the boundary before every word as a true positive, because every system keeps it."""
    tp = fp = fn = tn = 0
    exact = 0
    ed = []
    for w, gold, pred in pairs:
        g = boundaries(gold)
        n = len(w) - 1
        tp += len(g & pred)
        fp += len(pred - g)
        fn += len(g - pred)
        tn += n - len(g | pred)
        exact += int(g == pred)
        gs = '|'.join(gold)
        ps = mark(w, pred)
        ed.append(levenshtein(gs, ps) / max(len(gs), len(ps)))
    p, r, f = prf(tp, fp, fn)
    total = tp + fp + fn + tn
    words = len(pairs)
    pt, rt, ft = prf(tp + words, fp, fn)
    return {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'accuracy': (tp + tn) / total if total else 0.0,
            'precision': p, 'recall': r, 'f1': f, 'exact': exact / words if words else 0.0,
            'text_precision': pt, 'text_recall': rt, 'text_f1': ft,
            'text_accuracy': (tp + words + tn) / (total + words) if total else 0.0,
            'edit_distance': float(np.mean(ed)) if ed else 0.0, 'words': words, 'positions': total}


def mark(w, cuts):
    out = []
    for i, ch in enumerate(w):
        if i in cuts:
            out.append('|')
        out.append(ch)
    return ''.join(out)


def information_metrics(token_lists, n_chars):
    """Observed vocabulary, number of tokens, compression, bits per token (theoretical, empirical), information volume."""
    counter = Counter()
    n_tokens = 0
    for toks in token_lists:
        counter.update(toks)
        n_tokens += len(toks)
    v = len(counter)
    h = -sum(c / n_tokens * math.log2(c / n_tokens) for c in counter.values())
    return {'observed_vocab': v, 'tokens': n_tokens, 'characters': n_chars,
            'compression': n_chars / n_tokens, 'bits_theor': math.log2(v), 'bits_empir': h,
            'info_theor_mbit': n_tokens * math.log2(v) / 1e6, 'info_empir_mbit': n_tokens * h / 1e6,
            'efficiency': h / math.log2(v)}


def length_stats(lengths):
    a = np.array(lengths)
    return {'mean': float(a.mean()), 'median': float(np.median(a)), 'max': int(a.max())}
