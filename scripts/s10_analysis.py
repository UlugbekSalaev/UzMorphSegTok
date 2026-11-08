"""Step 9. Statistical analysis for the paper.

  * 95% confidence intervals by the sentence-level bootstrap (2,000 resamples) for the segmenters and tokenizers,
    and by the document-level bootstrap for the classification results;
  * paired differences between FEMSeg-uz and the other segmenters (bootstrap) and the exact McNemar test;
  * the stem boundary given by the lemma alone (independent of the CSE table) for every segmenter;
  * the share of words whose lemma-based stem boundary is in Omega(w) (an upper bound for constrained decoding);
  * results on words seen in the gold training words, seen only in the S or C words, and unseen;
  * error types of FEMSeg-uz and CSE-Seg+P;
  * five-fold cross-validation over all 684 sentences of UzUDT.
Result: results/analysis.json
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from scripts.s04_train_femseg import (build_sources, gold_split, fold_split, load_ref, ref_types, N_FOLDS, CONFIGS)
from uzseg.cse import read_rows, CSE, load_lexicon
from uzseg.metrics import boundaries, boundary_scores
from uzseg.rule_seg import Candidates

B_RESAMPLES = 2000
TEST = C.TEST_SETS[0]


def load(*path):
    p = os.path.join(C.RESULTS, *path)
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None


def mcnemar(n10, n01):
    n = n10 + n01
    if n == 0:
        return 1.0
    k = min(n10, n01)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def counts(sentences, pred):
    """Per sentence: TP, FP, FN, exact matches and reference words (pred: word -> boundary list)."""
    rows = []
    for s in sentences:
        tp = fp = fn = ex = 0
        for x in s['words']:
            g, p = boundaries(x['seg']), set(pred[x['w']])
            tp += len(g & p)
            fp += len(p - g)
            fn += len(g - p)
            ex += g == p
        rows.append((tp, fp, fn, ex, len(s['words'])))
    return np.array(rows, dtype=float)


def scores(c):
    """F1 and EM (%) from summed counts; c has the columns TP, FP, FN, EM, words (summed over the last axis -2)."""
    tp, fp, fn, ex, nw = (c[..., i] for i in range(5))
    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)
    return 100 * f1, 100 * ex / nw


def bootstrap(per_system, n_sent, seed=C.SEED):
    """per_system: {name: [count matrices, one per seed]} -> estimates, intervals and replicate values."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_sent, size=(B_RESAMPLES, n_sent))
    out, reps = {}, {}
    for name, mats in per_system.items():
        f1s, ems = [], []
        for m in mats:
            f1, em = scores(m[idx].sum(1))
            f1s.append(f1)
            ems.append(em)
        f1r, emr = np.mean(f1s, 0), np.mean(ems, 0)
        f1_full = float(np.mean([scores(m.sum(0))[0] for m in mats]))
        em_full = float(np.mean([scores(m.sum(0))[1] for m in mats]))
        out[name] = {'f1': f1_full, 'f1_ci': np.percentile(f1r, [2.5, 97.5]).tolist(),
                     'em': em_full, 'em_ci': np.percentile(emr, [2.5, 97.5]).tolist()}
        reps[name] = (f1r, emr)
    return out, reps


def diff(reps, a, b, full):
    d_f1 = reps[a][0] - reps[b][0]
    d_em = reps[a][1] - reps[b][1]
    return {'f1': full[a]['f1'] - full[b]['f1'], 'f1_ci': np.percentile(d_f1, [2.5, 97.5]).tolist(),
            'em': full[a]['em'] - full[b]['em'], 'em_ci': np.percentile(d_em, [2.5, 97.5]).tolist()}


def stem_accuracy(sentences, pred):
    """Share of word tokens whose predicted stem (up to the first boundary) equals the lemma-based stem."""
    total = good = total_out = good_out = 0
    for s in sentences:
        for x in s['stems']:
            p = pred[x['w']]
            k = min(p) if p else len(x['w'])
            ok = k == x['k']
            total += 1
            good += ok
            if not x['in_set']:
                total_out += 1
                good_out += ok
    return {'all': 100 * good / total, 'n': total, 'outside_set': 100 * good_out / max(total_out, 1),
            'n_outside': total_out}


def error_types(sentences, pred):
    c = Counter()
    examples = defaultdict(Counter)
    for s in sentences:
        for x in s['words']:
            g, p = boundaries(x['seg']), set(pred[x['w']])
            if g == p:
                continue
            gk = min(g) if g else len(x['w'])
            pk = min(p) if p else len(x['w'])
            kind = 'stem_too_long' if pk > gk else 'stem_too_short' if pk < gk else 'ending'
            c[kind] += 1
            mark = lambda cuts: ''.join(('|' if i in cuts else '') + ch for i, ch in enumerate(x['w']))
            examples[kind][(mark(g), mark(p))] += 1
    return dict(c), {k: [[a, b, n] for (a, b), n in v.most_common(8)] for k, v in examples.items()}


def feature_agreement(cse):
    """For the inflected reference words: does the selected CSE entry confirm each UD feature that has a counterpart in
    the table? Also the most frequent endings that are not in the table (words excluded from the reference)."""
    from uzseg.reference import UD_TO_CSE, agreement, stem_cut, word_lemma
    from uzseg.text import read_conllu
    per, missing = defaultdict(Counter), Counter()
    for fname in C.UD_FILES.values():
        for s in read_conllu(os.path.join(C.UD_DIR, fname)):
            for t in s['tokens']:
                pair = word_lemma(t)
                if pair is None or pair[0] == pair[1]:
                    continue
                w, lemma = pair
                k = stem_cut(w, lemma)
                if k is None or k >= len(w) or k < 1:
                    continue
                analyses = cse.analyses(w[:k], w[k:])
                if not analyses:
                    missing[w[k:]] += 1
                    continue
                options = {}
                for ei, morphs in analyses:
                    good, total = agreement(cse.entries[ei]['row'], t['feats'])
                    if morphs not in options or good > options[morphs][0]:
                        options[morphs] = (good, total, ei)
                row = cse.entries[max(options.items(), key=lambda x: (x[1][0], -len(x[0])))[1][2]]['row']
                for (key, val), (col, cval) in UD_TO_CSE.items():
                    if t['feats'].get(key) == val:
                        per[key]['total'] += 1
                        per[key]['good'] += row.get(col, '') == cval
                for ud, col in (('Person[psor]', 'possession'), ('Person', 'person')):
                    if t['feats'].get(ud):
                        per[ud]['total'] += 1
                        per[ud]['good'] += row.get(col, '')[:1] == t['feats'][ud]
    out = {k: {'total': v['total'], 'agree': 100 * v['good'] / v['total']} for k, v in per.items()}
    tot = sum(v['total'] for v in per.values())
    out['all'] = {'total': tot, 'agree': 100 * sum(v['good'] for v in per.values()) / tot}
    return out, missing.most_common(20)


def main():
    test = load_ref(TEST)
    n_sent = len(test)
    preds = load('segmenter_predictions.json')
    runs = defaultdict(list)
    for cfg in CONFIGS:
        for seed in C.SEEDS:
            r = load('femseg', f'{cfg}_s{seed}.json')
            if r:
                runs[cfg].append(r)
    result = {}
    # ------------------------------------------------------------------ confidence intervals and differences
    systems = {name: [counts(test, preds[name])] for name in ('Morfessor', 'Morfessor+UD', 'UzMorphAnalyser',
                                                              'CSE-Seg', 'CSE-Seg+P', 'CSE-Seg+P+UD')}
    for cfg, rs in runs.items():
        systems[f'FEMSeg-uz {cfg} hard'] = [counts(test, r['predictions']['hard']) for r in rs]
        systems[f'FEMSeg-uz {cfg} free'] = [counts(test, r['predictions']['free']) for r in rs]
    est, reps = bootstrap(systems, n_sent)
    result['segmenters_ci'] = est
    main_sys = 'FEMSeg-uz full hard'
    result['differences'] = {other: diff(reps, main_sys, other, est) for other in est if other != main_sys}
    # exact McNemar test on word tokens (one test per seed)
    tokens = [(x['w'], x['seg']) for s in test for x in s['words']]

    def correct(pred):
        return [boundaries(seg) == set(pred[w]) for w, seg in tokens]
    mc = {}
    for other in ('CSE-Seg+P', 'CSE-Seg+P+UD', 'Morfessor+UD', 'UzMorphAnalyser'):
        b = correct(preds[other])
        rows = []
        for r in runs['full']:
            a = correct(r['predictions']['hard'])
            n10 = sum(x and not y for x, y in zip(a, b))
            n01 = sum(y and not x for x, y in zip(a, b))
            rows.append({'seed': r['seed'], 'n10': n10, 'n01': n01, 'p': mcnemar(n10, n01)})
        mc[other] = rows
    result['mcnemar'] = mc
    # ------------------------------------------------------------------ stem boundary from the lemma alone
    stem = {name: stem_accuracy(test, preds[name]) for name in systems if name in preds}
    for cfg in ('full', 'v3v2'):
        if cfg in runs:
            vals = [stem_accuracy(test, r['predictions']['hard']) for r in runs[cfg]]
            stem[f'FEMSeg-uz {cfg} hard'] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
    cse = CSE(read_rows(C.CSE_CSV))
    lex = load_lexicon(C.LEXICON)
    cands = Candidates(cse, lex, min_stem=2)
    tot = inside = inside_full = n_full = 0
    for s in test:
        for x in s['stems']:
            cs = cands(x['w'])
            ks = {c['k'] for c in cs}
            tot += 1
            inside += x['k'] in ks
        for x in s['words']:
            g = boundaries(x['seg'])
            from uzseg.femseg import omega_cuts
            n_full += 1
            inside_full += g in omega_cuts(x['w'], cands(x['w']))
    stem['oracle_stem_in_omega'] = 100 * inside / tot
    stem['oracle_reference_in_omega'] = 100 * inside_full / n_full
    stem['omega_mean_size'] = float(np.mean([len(cands(x['w'])) for s in test for x in s['words']]))
    result['stem'] = stem
    result['features'], result['missing_endings'] = feature_agreement(cse)
    # ------------------------------------------------------------------ seen and unseen words
    sources, _ = build_sources(cse, lex)
    gold_types = set(ref_types(gold_split()[0])[0])
    sc_types = {w for w, _ in sources['silver']} | {w for w, _ in sources['syn']}
    groups = {'gold_seen': [], 'sc_seen': [], 'unseen': []}
    for w, seg in tokens:
        key = 'gold_seen' if w in gold_types else 'sc_seen' if w in sc_types else 'unseen'
        groups[key].append((w, seg))
    seen = {}
    for key, toks in groups.items():
        row = {'tokens': len(toks)}
        for name in ('CSE-Seg+P', 'CSE-Seg+P+UD', 'Morfessor+UD'):
            sc = boundary_scores([(w, seg, set(preds[name][w])) for w, seg in toks])
            row[name] = {'f1': 100 * sc['f1'], 'em': 100 * sc['exact']}
        vals = [boundary_scores([(w, seg, set(r['predictions']['hard'][w])) for w, seg in toks]) for r in runs['full']]
        row['FEMSeg-uz'] = {'f1': float(np.mean([100 * v['f1'] for v in vals])),
                            'em': float(np.mean([100 * v['exact'] for v in vals]))}
        a_rows = []
        for r in runs['full']:
            a = [boundaries(seg) == set(r['predictions']['hard'][w]) for w, seg in toks]
            b = [boundaries(seg) == set(preds['CSE-Seg+P'][w]) for w, seg in toks]
            n10 = sum(x and not y for x, y in zip(a, b))
            n01 = sum(y and not x for x, y in zip(a, b))
            a_rows.append({'seed': r['seed'], 'n10': n10, 'n01': n01, 'p': mcnemar(n10, n01)})
        row['mcnemar_vs_cse_seg_p'] = a_rows
        seen[key] = row
    result['seen_unseen'] = seen
    # ------------------------------------------------------------------ error types
    result['errors'] = {'CSE-Seg+P': error_types(test, preds['CSE-Seg+P'])}
    if runs['full']:
        result['errors']['FEMSeg-uz'] = error_types(test, runs['full'][0]['predictions']['hard'])
    # ------------------------------------------------------------------ cross-validation
    cv_runs = [load('femseg_cv', f'full_s{C.SEEDS[0]}_fold{k}.json') for k in range(N_FOLDS)]
    if all(cv_runs):
        all_sents, per = [], defaultdict(list)
        for k, r in enumerate(cv_runs):
            tr, _, te = fold_split(k)
            memory = ref_types(tr)[0]
            all_sents += te
            fem_pred = r['predictions']['hard']
            mem_pred = {w: sorted(boundaries(memory[w])) if w in memory else preds['CSE-Seg+P'][w]
                        for s in te for w in [x['w'] for x in s['words']]}
            per['FEMSeg-uz'].append(counts(te, fem_pred))
            per['CSE-Seg+P+UD'].append(counts(te, mem_pred))
            for name in ('CSE-Seg', 'CSE-Seg+P', 'Morfessor', 'UzMorphAnalyser'):
                per[name].append(counts(te, preds[name]))
        mats = {name: [np.concatenate(v)] for name, v in per.items()}
        est_cv, reps_cv = bootstrap(mats, len(all_sents))
        result['cv'] = {'sentences': len(all_sents), 'words': int(sum(len(s['words']) for s in all_sents)),
                        'scores': est_cv,
                        'differences': {o: diff(reps_cv, 'FEMSeg-uz', o, est_cv) for o in est_cv if o != 'FEMSeg-uz'},
                        'fold_f1': [r['scores']['hard'][f'fold{k}']['f1'] for k, r in enumerate(cv_runs)]}
    # ------------------------------------------------------------------ tokenizers
    tk = load('tokenizers_intrinsic.json') or {}
    rng = np.random.default_rng(C.SEED)
    tok_ci = {}
    if tk:
        n = len(next(iter(tk.values()))['boundary'][TEST]['per_sentence'])
        idx = rng.integers(0, n, size=(B_RESAMPLES, n))
        for name, r in tk.items():
            m = np.array(r['boundary'][TEST]['per_sentence'], dtype=float)
            s = m[idx].sum(1)
            tp, fp, fn, nw = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
            f1 = 100 * 2 * tp / np.maximum(2 * tp + fp + fn, 1)
            tf1 = 100 * 2 * (tp + nw) / (2 * (tp + nw) + fp + fn)
            tok_ci[name] = {'f1_ci': np.percentile(f1, [2.5, 97.5]).tolist(),
                            'text_f1_ci': np.percentile(tf1, [2.5, 97.5]).tolist()}
    result['tokenizers_ci'] = tok_ci
    # ------------------------------------------------------------------ classification
    ext = defaultdict(list)
    for f in sorted(os.listdir(os.path.join(C.RESULTS, 'extrinsic'))) if os.path.isdir(
            os.path.join(C.RESULTS, 'extrinsic')) else []:
        if f.endswith('.json'):
            r = load('extrinsic', f)
            ext[r['tokenizer']].append(r)
    cls = {}
    if ext:
        any_r = next(iter(ext.values()))[0]
        gold = np.array(any_r['test_gold'])
        n, k = len(gold), len(any_r['labels'])
        idx = rng.integers(0, n, size=(1000, n))

        def macro(pred_rep, gold_rep):
            f = []
            for c in range(k):
                tp = ((pred_rep == c) & (gold_rep == c)).sum(1)
                fp = ((pred_rep == c) & (gold_rep != c)).sum(1)
                fn = ((pred_rep != c) & (gold_rep == c)).sum(1)
                f.append(np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0))
            return 100 * np.mean(f, 0)
        g_rep = gold[idx]
        rep = {}
        for name, rs in ext.items():
            if len(rs) < len(C.SEEDS):
                continue
            vals = np.mean([macro(np.array(r['test_pred'])[idx], g_rep) for r in rs], 0)
            rep[name] = vals
            cls[name] = {'macro_f1_ci': np.percentile(vals, [2.5, 97.5]).tolist()}
        if 'bpe_raw_32000' in rep:
            for name in rep:
                d = rep[name] - rep['bpe_raw_32000']
                cls[name]['diff_vs_bpe_raw_ci'] = np.percentile(d, [2.5, 97.5]).tolist()
    result['classification_ci'] = cls
    json.dump(result, open(os.path.join(C.RESULTS, 'analysis.json'), 'w', encoding='utf-8'), indent=1,
              ensure_ascii=False)
    for name, e in result['segmenters_ci'].items():
        print(f"{name:28s} F1 {e['f1']:.2f} [{e['f1_ci'][0]:.2f}, {e['f1_ci'][1]:.2f}]  "
              f"EM {e['em']:.2f} [{e['em_ci'][0]:.2f}, {e['em_ci'][1]:.2f}]")
    print('stem', {k: (round(v['all'], 2) if isinstance(v, dict) else round(v, 2)) for k, v in stem.items()})
    if 'cv' in result:
        print('cv', {k: round(v['f1'], 2) for k, v in result['cv']['scores'].items()})


if __name__ == '__main__':
    main()
