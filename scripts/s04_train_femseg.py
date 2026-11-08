"""Step 4. Train and evaluate the neural segmenter FEMSeg-uz.

Usage:  python scripts/s04_train_femseg.py --config full --seed 17
        python scripts/s04_train_femseg.py --config full --seed 17 --fold 0     (five-fold cross-validation)

Configurations (training sources and input channels):
  v2            S: synthetic CSE word forms (lexicon stem + CSE chain);
  v3            C: corpus words of 50,000 sentences segmented by CSE-Seg+P;
  v3v2          C + S;
  full          C + S + G: 80% of the reference words of the UzUDT train part (main model);
  full_nocse    full without the CSE channel;  full_char  full with characters only;
  full_nocnn, full_notrans   full without the convolutions or without the Transformer layer.
Development data: the other 20% of the sentences of the UzUDT train part.
The one-cycle schedule always runs for all epochs, because the learning rate becomes small only in the last epochs; the
parameters of the epoch with the highest development F1 (hard decoding) are kept.
In cross-validation, all 684 sentences of UzUDT are divided into five folds; one fold is the test part, and the others
are divided into G (80%) and development data (20%). Only predictions are kept, not the models.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from uzseg.cse import read_rows, CSE, load_lexicon
from uzseg.femseg import FEMSegUz, CharVocab, encode_word, collate, seg_to_tags, omega_cuts, cuts_to_tags, predict_cuts
from uzseg.metrics import boundary_scores
from uzseg.rule_seg import Candidates
from uzseg.synthetic import synthetic_words
from uzseg.text import read_conllu, normalize, strip_infinitive

CONFIGS = {
    'v2': dict(sources=('syn',)),
    'v3': dict(sources=('silver',)),
    'v3v2': dict(sources=('silver', 'syn')),
    'full': dict(sources=('silver', 'syn', 'gold')),
    'full_nocse': dict(sources=('silver', 'syn', 'gold'), channels=('phon', 'digr', 'syll')),
    'full_char': dict(sources=('silver', 'syn', 'gold'), channels=()),
    'full_nocnn': dict(sources=('silver', 'syn', 'gold'), use_cnn=False),
    'full_notrans': dict(sources=('silver', 'syn', 'gold'), use_transformer=False),
}
GOLD_REPEAT = 10
N_FOLDS = 5


def load_ref(name):
    return [json.loads(line) for line in open(os.path.join(C.REF_DIR, name + '.jsonl'), encoding='utf-8')]


def ref_types(sentences):
    """Reference word types (the first segmentation of a word) and the list of word tokens."""
    tokens = [(w['w'], tuple(w['seg'])) for s in sentences for w in s['words']]
    types = {}
    for w, seg in tokens:
        types.setdefault(w, seg)
    return types, tokens


def split_80_20(sents, seed):
    rng = random.Random(seed)
    idx = list(range(len(sents)))
    rng.shuffle(idx)
    n_dev = len(idx) // 5
    return [sents[i] for i in sorted(idx[n_dev:])], [sents[i] for i in sorted(idx[:n_dev])]


def gold_split():
    """Gold training sentences and development sentences of the official train part."""
    return split_80_20(load_ref(C.DEV_SET), C.SEED)


def fold_split(fold):
    """Cross-validation: (gold training sentences, development sentences, test sentences) of one fold."""
    sents = load_ref('UzUDT_train') + load_ref('UzUDT_test')
    rng = random.Random(C.SEED)
    idx = list(range(len(sents)))
    rng.shuffle(idx)
    test_idx = set(idx[fold::N_FOLDS])
    train, dev = split_80_20([s for i, s in enumerate(sents) if i not in test_idx], C.SEED + fold + 1)
    return train, dev, [sents[i] for i in sorted(test_idx)]


def eval_lemmas():
    out = set()
    for f in C.UD_FILES.values():
        for s in read_conllu(os.path.join(C.UD_DIR, f)):
            for t in s['tokens']:
                lem = normalize(t['lemma'])
                out.update({lem, strip_infinitive(lem)})
    return out


def build_sources(cse, lex):
    silver = []
    for line in open(os.path.join(C.CORPUS_DIR, 'silver_words.tsv'), encoding='utf-8'):
        w, _, seg = line.rstrip('\n').split('\t')
        silver.append((w, tuple(seg.split('|'))))
    syn = [(w, seg) for w, seg, _ in synthetic_words(cse, lex, eval_lemmas(), n_noun=300, n_verb=120)]
    rng = random.Random(C.SEED)
    rng.shuffle(silver)
    n_hold = len(silver) // 20
    holdout = {'silver': silver[:n_hold], 'syn': syn[:len(syn) // 20]}
    return {'silver': silver[n_hold:], 'syn': syn[len(syn) // 20:]}, holdout


def make_batches(items, size, rng):
    """Batches of words of similar length (built once; their order is shuffled in every epoch)."""
    order = list(range(len(items)))
    rng.shuffle(order)
    order.sort(key=lambda i: len(items[i][0]['c']))
    return [collate([items[i] for i in order[j:j + size]]) for j in range(0, len(order), size)]


def score_tokens(model, tokens, vocab, cands, mode):
    words = sorted({w for w, _ in tokens})
    pred = dict(zip(words, predict_cuts(model, words, vocab, cands, mode)))
    return boundary_scores([(w, seg, pred[w]) for w, seg in tokens])


def evaluate_all(model, vocab, cands, holdout, dev_tokens, test_sets):
    """Scores of a trained model: development and test (free, hard), held-out words of sources S and C, predictions
    for the test words and the speed."""
    model.eval()
    res = {'scores': {}}
    for mode in ('free', 'hard'):
        res['scores'][mode] = {'dev': score_tokens(model, dev_tokens, vocab, cands, mode)}
        for name, sents in test_sets.items():
            res['scores'][mode][name] = score_tokens(model, ref_types(sents)[1], vocab, cands, mode)
    res['in_domain'] = {}
    for src, items in holdout.items():
        words = [w for w, _ in items]
        pred = predict_cuts(model, words, vocab, cands, 'free')
        sc = boundary_scores([(w, seg, p) for (w, seg), p in zip(items, pred)])
        correct = total = 0
        for (w, seg), p in zip(items, pred):
            g, q = seg_to_tags(seg), cuts_to_tags(len(w), p)
            correct += sum(a == b for a, b in zip(g, q))
            total += len(g)
        sc['tag_accuracy'] = correct / total
        res['in_domain'][src] = sc
    test_words = set()
    for sents in test_sets.values():
        test_words |= {w for w, _ in ref_types(sents)[1]} | {x['w'] for s in sents for x in s['stems']}
    test_words = sorted(test_words)
    res['predictions'] = {}
    t = time.time()
    for mode in ('free', 'hard'):
        pred = predict_cuts(model, test_words, vocab, cands, mode)
        res['predictions'][mode] = {w: sorted(p) for w, p in zip(test_words, pred)}
        if mode == 'hard':
            res['speed_words_per_s'] = len(test_words) / (time.time() - t)
        t = time.time()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='full')
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--fold', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--threads', type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = CONFIGS[args.config]
    name = f'{args.config}_s{args.seed}' + (f'_fold{args.fold}' if args.fold is not None else '')
    out_dir = os.path.join(C.RESULTS, 'femseg_cv' if args.fold is not None else 'femseg')
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, name + '.json')):
        print('done already:', name)
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    t0 = time.time()

    cse = CSE(read_rows(C.CSE_CSV))
    lex = load_lexicon(C.LEXICON)
    cands = Candidates(cse, lex, min_stem=2)
    sources, holdout = build_sources(cse, lex)
    if args.fold is None:
        gold_train, gold_dev = gold_split()
        test_sets = {ts: load_ref(ts) for ts in C.TEST_SETS}
    else:
        gold_train, gold_dev, fold_test = fold_split(args.fold)
        test_sets = {f'fold{args.fold}': fold_test}
    gold_types, _ = ref_types(gold_train)
    _, dev_tokens = ref_types(gold_dev)
    train = []
    for src in cfg['sources']:
        if src == 'gold':
            train += list(gold_types.items()) * GOLD_REPEAT
        else:
            train += sources[src]
    vocab = CharVocab([w for w, _ in train] + [w for w, _ in sources['silver']])
    data = [(encode_word(w, vocab, cands(w)), seg_to_tags(seg)) for w, seg in train]
    print(f'{name}: {len(data)} training words ({time.time() - t0:.0f} s)', flush=True)

    model = FEMSegUz(len(vocab.itos), channels=cfg.get('channels', ('phon', 'digr', 'syll', 'cse')),
                     use_cnn=cfg.get('use_cnn', True), use_transformer=cfg.get('use_transformer', True))
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(data) + args.batch - 1) // args.batch) + 1
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    rng = random.Random(args.seed)
    all_batches = make_batches(data, args.batch, rng)
    best_f1, best_state, history, best_epoch = -1.0, None, [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        te, total, nb = time.time(), 0.0, 0
        rng.shuffle(all_batches)
        for batch in all_batches:
            loss = model.loss(batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += loss.item()
            nb += 1
        dev_free = score_tokens(model, dev_tokens, vocab, cands, 'free')
        dev_hard = score_tokens(model, dev_tokens, vocab, cands, 'hard')
        history.append({'epoch': epoch, 'loss': total / nb, 'dev_f1_free': dev_free['f1'],
                        'dev_f1_hard': dev_hard['f1'], 'time': time.time() - te})
        print(f"  epoch {epoch}: loss {total / nb:.4f}, dev F1 free {dev_free['f1']:.4f} hard "
              f"{dev_hard['f1']:.4f} ({time.time() - te:.0f} s)", flush=True)
        if dev_hard['f1'] > best_f1:
            best_f1, best_epoch = dev_hard['f1'], epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    train_time = sum(h['time'] for h in history)

    result = {'config': args.config, 'seed': args.seed, 'fold': args.fold, 'params': n_params,
              'train_words': len(data), 'train_time_s': train_time, 'history': history, 'best_epoch': best_epoch}
    result.update(evaluate_all(model, vocab, cands, holdout, dev_tokens, test_sets))
    if args.fold is None:
        torch.save({'state': model.state_dict(), 'vocab': vocab.itos, 'config': args.config,
                    'channels': sorted(model.channels), 'use_cnn': model.use_cnn,
                    'use_transformer': model.use_transformer}, os.path.join(C.SEG_DIR, name + '.pt'))
    json.dump(result, open(os.path.join(out_dir, name + '.json'), 'w'), indent=1)
    first = next(iter(test_sets))
    print(f"{name}: done ({time.time() - t0:.0f} s); {first} hard F1 {result['scores']['hard'][first]['f1']:.4f}",
          flush=True)


if __name__ == '__main__':
    main()
