"""Step 5. Evaluate the segmenters on the reference set.

Systems:
  Morfessor          unsupervised statistical segmenter, trained on the corpus word types; the count dampening
                     (none, log, ones) and the corpus weight are selected on the development words;
  Morfessor+UD       semi-supervised Morfessor: the gold words of FEMSeg-uz (80% of the UzUDT train part) are
                     annotations, the corpus weight is tuned on the other 20%;
  UzMorphAnalyser    the earlier rule-based analyser (stem + CSE ending), package version 0.0.22;
  CSE-Seg            CSE table + lexicon, longest ending;
  CSE-Seg+P          CSE table + lexicon + paradigm statistics of the corpus;
  CSE-Seg+P+UD       CSE-Seg+P, but words of the gold training set receive their reference segmentation;
  FEMSeg-uz          from results/femseg/*.json (step 4).
Results: results/segmenters.json and results/segmenter_predictions.json (boundary sets of the evaluated words).
Existing results are kept; only missing systems are computed.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from uzseg.metrics import boundary_scores, boundaries
from uzseg.text import normalize
from scripts.s02_segment_corpus import load_segmenter
from scripts.s04_train_femseg import gold_split, ref_types, load_ref, fold_split, N_FOLDS

DAMPENING = {'none': None, 'log': lambda x: int(round(math.log(x + 1, 2))), 'ones': lambda x: 1}


def train_morfessor(dampening, min_freq=3, annotations=None, devel=None, name=None):
    import morfessor
    io = morfessor.MorfessorIO()
    path = os.path.join(C.MODELS, name or f'morfessor_{dampening}.bin')
    if os.path.exists(path):
        return io.read_binary_model_file(path)
    data = []
    for line in open(os.path.join(C.CORPUS_DIR, 'word_freq.tsv'), encoding='utf-8'):
        w, c = line.rstrip('\n').split('\t')
        if int(c) >= min_freq:
            data.append((int(c), w))
    model = morfessor.BaselineModel()
    model.load_data(data, count_modifier=DAMPENING[dampening])
    if annotations:
        model.set_annotations(annotations)
    # the corpus weight is changed until boundary precision and recall on the development words are equal
    model.set_corpus_weight_updater(morfessor.baseline.AnnotationCorpusWeight(devel, threshold=0.01))
    model.train_batch()
    io.write_binary_model_file(path, model)
    return model


def dev_devel():
    return {w: [list(seg)] for w, seg in ref_types(gold_split()[1])[0].items()}


def uzmorph_segment(analyser, cse, w):
    """UzMorphAnalyser output -> morphemes (the ending is divided with the CSE table)."""
    try:
        res = analyser.analyze(w)
    except Exception:
        return (w,)
    if not res:
        return (w,)
    a = res[0]
    affixed = normalize(a.get('affixed', '') or '')
    if not affixed or not w.endswith(affixed) or len(affixed) >= len(w):
        return (w,)
    stem = w[:-len(affixed)]
    analyses = cse.analyses(stem, affixed)
    same_id = [m for ei, m in analyses if str(cse.entries[ei]['id']) == str(a.get('id'))]
    morphs = same_id[0] if same_id else analyses[0][1] if analyses else (affixed,)
    return (stem,) + tuple(morphs)


def evaluate(fn, sets, extra_words, store):
    out = {}
    for name, toks in sets.items():
        words = {w for w, _ in toks}
        pred = {w: boundaries(fn(w)) for w in words}
        out[name] = boundary_scores([(w, seg, pred[w]) for w, seg in toks])
        store.update({w: sorted(c) for w, c in pred.items()})
    for w in extra_words - set(store):
        store[w] = sorted(boundaries(fn(w)))
    return out


def main():
    test_sents = {ts: load_ref(ts) for ts in C.TEST_SETS}
    train_sents, dev_sents = gold_split()
    sets = {ts: ref_types(s)[1] for ts, s in test_sents.items()}
    sets['dev'] = ref_types(dev_sents)[1]
    folds = [fold_split(k)[2] for k in range(N_FOLDS)]
    extra = {x['w'] for ss in list(test_sents.values()) + [dev_sents] + folds for s in ss for x in s['stems']}
    extra |= {w for f in folds for w, _ in ref_types(f)[1]}
    seg_p, seg_l, cse, lex = load_segmenter()
    memory = ref_types(train_sents)[0]
    res_path = os.path.join(C.RESULTS, 'segmenters.json')
    pred_path = os.path.join(C.RESULTS, 'segmenter_predictions.json')
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    preds = json.load(open(pred_path, encoding='utf-8')) if os.path.exists(pred_path) else {}

    def morf(model):
        return lambda w: tuple(model.viterbi_segment(w)[0])

    # Morfessor: the dampening is chosen by the development F1
    grid = results.get('Morfessor_grid', {})
    for d in DAMPENING:
        if d not in grid:
            m = morf(train_morfessor(d, devel=dev_devel()))
            grid[d] = boundary_scores([(w, seg, boundaries(m(w))) for w, seg in sets['dev']])['f1']
            results['Morfessor_grid'] = grid
            json.dump(results, open(res_path, 'w'), indent=1)
    best_d = max(grid, key=grid.get)
    print('Morfessor dampening (dev F1):', grid, '->', best_d, flush=True)

    def uzma():
        from UzMorphAnalyser import UzMorphAnalyser
        ua = UzMorphAnalyser()
        return lambda w: uzmorph_segment(ua, cse, w)

    def morf_ud():
        annotations = {w: [seg] for w, seg in memory.items()}
        return morf(train_morfessor(best_d, annotations=annotations, devel=dev_devel(), name='morfessor_ud.bin'))

    systems = {'CSE-Seg': lambda: seg_l.segment, 'CSE-Seg+P': lambda: seg_p.segment,
               'CSE-Seg+P+UD': lambda: (lambda w: memory.get(w) or seg_p.segment(w)),
               'Morfessor': lambda: morf(train_morfessor(best_d, devel=dev_devel())), 'Morfessor+UD': morf_ud,
               'UzMorphAnalyser': uzma}
    for name, make in systems.items():
        if name in results and name in preds:
            continue
        preds[name] = {}
        results[name] = evaluate(make(), sets, extra, preds[name])
        results[name]['dampening'] = best_d if name.startswith('Morfessor') else None
        json.dump(results, open(res_path, 'w'), indent=1)
        json.dump(preds, open(pred_path, 'w', encoding='utf-8'), ensure_ascii=False)
    for name in systems:
        r = results[name][C.TEST_SETS[0]]
        print(f"{name:16s} P {r['precision']:.4f} R {r['recall']:.4f} F1 {r['f1']:.4f}  EM {r['exact']:.4f}  "
              f"dev F1 {results[name]['dev']['f1']:.4f}")


if __name__ == '__main__':
    main()
