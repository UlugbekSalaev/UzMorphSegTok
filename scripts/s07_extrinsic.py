"""Step 8a. News classification by topic (nine categories).

For every tokenizer, the same small Transformer classifier is trained from scratch (2 layers, d = 128, 4 heads).
Only the tokenizer changes, so differences in accuracy, macro-F1 and cost come from tokenization.
Every tokenizer is trained with three seeds.

Usage:  python scripts/s07_extrinsic.py [tokenizer names]
Result: results/extrinsic/<tokenizer>_s<seed>.json (with the predicted label of every test document)
"""
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from uzseg.tokenizers import load_tokenizers

# '+seg': words are segmented before encoding (*_cse_*: CSE-Seg+P, *_fem_*: FEMSeg-uz)
MAIN_TOKENIZERS = ['bpe_raw_32000', 'uni_raw_32000', 'bpe_cse_32000', 'uni_cse_32000',
                   'bpe_cse_32000+seg', 'uni_cse_32000+seg', 'bpe_fem_32000+seg', 'uni_fem_32000+seg',
                   'mT5', 'XLM-R', 'mBERT']
MAX_LEN, D, LAYERS, HEADS, EPOCHS, BATCH, LR = 512, 128, 2, 4, 8, 32, 5e-4


class Classifier(nn.Module):
    def __init__(self, vocab, n_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab, D, padding_idx=0)
        self.pos = nn.Embedding(MAX_LEN, D)
        layer = nn.TransformerEncoderLayer(D, HEADS, 2 * D, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.out = nn.Linear(D, n_classes)

    def forward(self, x, mask):
        pos = torch.arange(x.shape[1]).unsqueeze(0)
        h = self.enc(self.emb(x) + self.pos(pos), src_key_padding_mask=~mask)
        h = (h * mask.unsqueeze(2)).sum(1) / mask.sum(1, keepdim=True)
        return self.out(h)


def load_split(name):
    return [json.loads(line) for line in open(os.path.join(C.EXT_DIR, f'uztc_{name}.jsonl'), encoding='utf-8')]


def load_data(large):
    """Main experiment: 2,000 training documents per category. With large=True (check with 5,000 documents per
    category), the reserve is added to the training part."""
    data = {k: load_split(k) for k in ('train', 'dev', 'test')}
    if large:
        data['train'] = data['train'] + load_split('reserve')
    return data


def encode_all(tok, data):
    seg = getattr(tok, 'segmenter', None)
    if seg is not None:                     # all words are segmented once beforehand
        from uzseg.text import word_spans, normalize, is_word
        seg.prefetch({w for x in data for a, b in word_spans(x['text'])
                      for w in [normalize(x['text'][a:b])] if is_word(w)})
    return [tok.ids(x['text']) for x in data]


def batches(seqs, labels, rng, shuffle):
    """Batches of documents of similar length: little padding, so the cost follows the number of tokens."""
    order = list(range(len(seqs)))
    if shuffle:
        rng.shuffle(order)
    out = []
    for i in range(0, len(order), BATCH * 20):
        chunk = sorted(order[i:i + BATCH * 20], key=lambda j: len(seqs[j]))
        out += [chunk[k:k + BATCH] for k in range(0, len(chunk), BATCH)]
    if shuffle:
        rng.shuffle(out)
    for idx in out:
        L = max(len(seqs[j]) for j in idx)
        x = torch.zeros(len(idx), L, dtype=torch.long)
        for r, j in enumerate(idx):
            x[r, :len(seqs[j])] = torch.tensor(seqs[j])
        yield idx, x, x != 0, torch.tensor([labels[j] for j in idx])


def macro_f1(gold, pred, n):
    f = []
    for c in range(n):
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f))


def predict(model, seqs, labels):
    model.eval()
    pred = [None] * len(seqs)
    with torch.no_grad():
        for idx, x, m, y in batches(seqs, labels, None, False):
            for j, p in zip(idx, model(x, m).argmax(1).tolist()):
                pred[j] = p
    return pred


def evaluate(model, seqs, labels, n):
    pred = predict(model, seqs, labels)
    acc = sum(g == p for g, p in zip(labels, pred)) / len(labels)
    return acc, macro_f1(labels, pred, n), pred


def run(tok, seed, data, threads):
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    labels = sorted({x['label'] for x in data['train']})
    lab = {l: i for i, l in enumerate(labels)}
    raw = {k: encode_all(tok, v) for k, v in data.items()}
    longest = max(len(s) for v in raw.values() for s in v)
    if longest > MAX_LEN:
        sys.exit(f'{tok.name}: a document has {longest} tokens, more than {MAX_LEN}')
    counts = Counter(t for s in raw['train'] for t in s)
    remap = {t: i + 2 for i, (t, _) in enumerate(counts.most_common())}      # 0: padding, 1: unknown token
    seqs = {k: [[remap.get(t, 1) for t in s] for s in v] for k, v in raw.items()}
    ys = {k: [lab[x['label']] for x in v] for k, v in data.items()}
    model = Classifier(len(remap) + 2, len(labels))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    rng = random.Random(seed)
    best, best_state, history, train_time = -1, None, [], 0.0
    padded = attn = 0
    for epoch in range(EPOCHS):
        model.train()
        t = time.time()
        for _, x, m, y in batches(seqs['train'], ys['train'], rng, True):
            loss = nn.functional.cross_entropy(model(x, m), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            padded += x.numel()
            attn += x.shape[0] * x.shape[1] ** 2
        train_time += time.time() - t
        acc, f1, _ = evaluate(model, seqs['dev'], ys['dev'], len(labels))
        history.append({'epoch': epoch + 1, 'dev_acc': acc, 'dev_f1': f1, 'time': time.time() - t})
        if f1 > best:
            best, best_epoch = f1, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    acc, f1, pred = evaluate(model, seqs['test'], ys['test'], len(labels))
    lens = [len(s) for s in raw['train']]
    return {'tokenizer': tok.name, 'seed': seed, 'test_acc': acc, 'test_macro_f1': f1, 'train_time_s': train_time,
            'history': history, 'mean_tokens': float(np.mean(lens)), 'max_tokens': longest,
            'observed_vocab': len(remap), 'padded_positions': padded, 'attention_cost': attn,
            'params': sum(p.numel() for p in model.parameters()), 'best_epoch': best_epoch, 'epochs': EPOCHS,
            'labels': labels, 'test_gold': ys['test'], 'test_pred': pred,
            'test_ids': [x['id'] for x in data['test']]}


def main():
    """Several processes can run at the same time: a (tokenizer, seed) job is reserved by a .lock file.
    ORDER=reverse takes the jobs in reverse order (from the last seed).
    --large: the check with 5,000 training documents per category (seed 17 only, results/extrinsic_5000)."""
    args = [a for a in sys.argv[1:] if a != '--large']
    large = '--large' in sys.argv[1:]
    names = args or MAIN_TOKENIZERS
    threads = int(os.environ.get('THREADS', '4'))
    out_dir = os.path.join(C.RESULTS, 'extrinsic_5000' if large else 'extrinsic')
    os.makedirs(out_dir, exist_ok=True)
    data = load_data(large)
    toks = load_tokenizers(C.TOK_DIR, names)
    jobs = [(seed, name) for seed in (C.SEEDS[:1] if large else C.SEEDS) for name in names]
    if os.environ.get('ORDER') == 'reverse':
        jobs = jobs[::-1]
    for seed, name in jobs:
        path = os.path.join(out_dir, f'{name}_s{seed}.json')
        lock = path[:-5] + '.lock'
        if os.path.exists(path) or name not in toks:
            continue
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        r = run(toks[name], seed, data, threads)
        json.dump(r, open(path, 'w'), indent=1)
        os.remove(lock)
        print(f"{name:18s} s{seed}: acc {r['test_acc']:.4f}  F1 {r['test_macro_f1']:.4f}  "
              f"time {r['train_time_s']:.0f} s  tokens {r['mean_tokens']:.0f}", flush=True)


if __name__ == '__main__':
    main()
