"""FEMSeg-uz: a character-level neural morphological segmenter for Uzbek.

Every character of a word w = c_1 ... c_n receives a BMES tag (B: begin of a morpheme, M: middle, E: end,
S: single-character morpheme). The input vector of a character joins five embeddings:
    x_t = [E_c(c_t); E_p(phon_t); E_d(digr_t); E_s(syll_t); E_k(kappa_t)],
where kappa_t is the CSE channel: is there a morpheme boundary before position t in some candidate of Omega(w)
(0: no, 1: yes, 2: yes, in a candidate with lexical support). Then:
    convolutions (widths 3, 5, 7) -> BiLSTM -> Transformer layer -> linear layer -> CRF.

Decoding:
    free: the Viterbi algorithm (the best tag sequence of the CRF);
    hard: the segmentation of Omega(w) with the highest CRF score (CSE-constrained decoding).
"""
import math

import torch
import torch.nn as nn

from .phonology import char_features

TAGS = ['B', 'M', 'E', 'S']
B, M, E, S = range(4)
# BMES scheme: which tag can follow which tag
ALLOWED = {(B, M), (B, E), (M, M), (M, E), (E, B), (E, S), (S, B), (S, S)}
START_OK, END_OK = {B, S}, {E, S}


def seg_to_tags(seg):
    tags = []
    for m in seg:
        tags += [S] if len(m) == 1 else [B] + [M] * (len(m) - 2) + [E]
    return tags


def tags_to_cuts(tags):
    """Tags -> set of boundaries (after a morpheme that ends with E or S)."""
    return {i + 1 for i, t in enumerate(tags[:-1]) if t in (E, S)}


def cuts_to_tags(n, cuts):
    seg, last = [], 0
    for c in sorted(cuts) + [n]:
        seg.append(c - last)
        last = c
    tags = []
    for L in seg:
        tags += [S] if L == 1 else [B] + [M] * (L - 2) + [E]
    return tags


class CharVocab:
    def __init__(self, words):
        chars = sorted({ch for w in words for ch in w})
        self.itos = ['<pad>', '<unk>'] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def encode(self, w):
        return [self.stoi.get(ch, 1) for ch in w]


def cse_channel(w, cand_list):
    """kappa_t: 0 no boundary, 1 a boundary in some candidate, 2 a boundary in a candidate with lexical support."""
    k = [0] * len(w)
    for c in cand_list:
        if c['k'] == len(w):
            continue
        pos = c['k']
        cuts = [pos]
        for m in c['morphs'][:-1]:
            pos += len(m)
            cuts.append(pos)
        for p in cuts:
            if 0 < p < len(w):
                k[p] = max(k[p], 2 if c['lex'] else 1)
    return k


def omega_cuts(w, cand_list):
    """All segmentations of Omega(w) as boundary sets, without repetition."""
    out = {frozenset()}
    for c in cand_list:
        if c['k'] == len(w):
            continue
        pos = c['k']
        cuts = {pos}
        for m in c['morphs'][:-1]:
            pos += len(m)
            cuts.add(pos)
        out.add(frozenset(cuts))
    return [set(x) for x in out]


def encode_word(w, vocab, cand_list):
    feats = char_features(w)
    return {'c': vocab.encode(w), 'p': [f[0] for f in feats], 'd': [f[1] for f in feats],
            's': [f[2] for f in feats], 'k': cse_channel(w, cand_list) if cand_list is not None else [0] * len(w)}


def collate(items):
    """items: [(encoded word, tags or None)] -> tensors."""
    n = max(len(x[0]['c']) for x in items)
    bsz = len(items)
    out = {key: torch.zeros(bsz, n, dtype=torch.long) for key in ('c', 'p', 'd', 's', 'k', 'y')}
    mask = torch.zeros(bsz, n, dtype=torch.bool)
    for i, (enc, tags) in enumerate(items):
        L = len(enc['c'])
        for key in ('c', 'p', 'd', 's', 'k'):
            out[key][i, :L] = torch.tensor(enc[key])
        if tags is not None:
            out['y'][i, :L] = torch.tensor(tags)
        mask[i, :L] = True
    out['mask'] = mask
    return out


class CRF(nn.Module):
    """Linear-chain CRF; transitions that break the BMES scheme are closed with a penalty of -1e4."""

    def __init__(self, n=4):
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(n, n))
        self.start = nn.Parameter(torch.zeros(n))
        self.end = nn.Parameter(torch.zeros(n))
        mask = torch.full((n, n), -1e4)
        for a, b in ALLOWED:
            mask[a, b] = 0.0
        smask = torch.full((n,), -1e4)
        emask = torch.full((n,), -1e4)
        for t in START_OK:
            smask[t] = 0.0
        for t in END_OK:
            emask[t] = 0.0
        self.register_buffer('tmask', mask)
        self.register_buffer('smask', smask)
        self.register_buffer('emask', emask)

    def params(self):
        return self.trans + self.tmask, self.start + self.smask, self.end + self.emask

    def nll(self, em, y, mask):
        A, st, en = self.params()
        bsz, n, _ = em.shape
        ar = torch.arange(bsz)
        lengths = mask.sum(1)
        score = st[y[:, 0]] + em[ar, 0, y[:, 0]]
        for t in range(1, n):
            m = mask[:, t].float()
            score = score + (em[ar, t, y[:, t]] + A[y[:, t - 1], y[:, t]]) * m
        score = score + en[y[ar, lengths - 1]]
        alpha = st.unsqueeze(0) + em[:, 0]
        for t in range(1, n):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + A.unsqueeze(0) + em[:, t].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), nxt, alpha)
        log_z = torch.logsumexp(alpha + en.unsqueeze(0), dim=1)
        return (log_z - score).mean()

    def viterbi(self, em, mask):
        A, st, en = self.params()
        bsz, n, _ = em.shape
        score = st.unsqueeze(0) + em[:, 0]
        back = []
        for t in range(1, n):
            s, idx = (score.unsqueeze(2) + A.unsqueeze(0)).max(dim=1)
            s = s + em[:, t]
            score = torch.where(mask[:, t].unsqueeze(1), s, score)
            back.append(idx)
        score = score + en.unsqueeze(0)
        best_score, last = score.max(dim=1)
        lengths = mask.sum(1).tolist()
        paths = []
        for b in range(bsz):
            L = lengths[b]
            tag = last[b].item()
            path = [tag]
            for t in range(L - 2, -1, -1):
                tag = back[t][b, tag].item()
                path.append(tag)
            paths.append(path[::-1])
        return paths, best_score


class FEMSegUz(nn.Module):
    def __init__(self, n_chars, channels=('phon', 'digr', 'syll', 'cse'), d_char=48, d_feat=8, n_filters=48,
                 hidden=64, use_cnn=True, use_transformer=True, dropout=0.2):
        super().__init__()
        self.channels = set(channels)
        self.use_cnn, self.use_transformer = use_cnn, use_transformer
        self.emb_c = nn.Embedding(n_chars, d_char, padding_idx=0)
        d_in = d_char
        self.emb = nn.ModuleDict()
        sizes = {'phon': 6, 'digr': 3, 'syll': 4, 'cse': 3}
        for ch in ('phon', 'digr', 'syll', 'cse'):
            if ch in self.channels:
                self.emb[ch] = nn.Embedding(sizes[ch], d_feat)
                d_in += d_feat
        self.drop = nn.Dropout(dropout)
        if use_cnn:
            self.convs = nn.ModuleList([nn.Conv1d(d_in, n_filters, k, padding=k // 2) for k in (3, 5, 7)])
            d_in = 3 * n_filters
        self.lstm = nn.LSTM(d_in, hidden, batch_first=True, bidirectional=True)
        if use_transformer:
            layer = nn.TransformerEncoderLayer(2 * hidden, 4, 4 * hidden, dropout=0.1, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.out = nn.Linear(2 * hidden, 4)
        self.crf = CRF(4)

    def emissions(self, batch):
        x = [self.emb_c(batch['c'])]
        for ch, key in (('phon', 'p'), ('digr', 'd'), ('syll', 's'), ('cse', 'k')):
            if ch in self.channels:
                x.append(self.emb[ch](batch[key]))
        x = self.drop(torch.cat(x, dim=-1))
        if self.use_cnn:
            h = x.transpose(1, 2)
            x = torch.cat([torch.relu(conv(h)) for conv in self.convs], dim=1).transpose(1, 2)
        lengths = batch['mask'].sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=x.shape[1])
        if self.use_transformer:
            h = self.transformer(h, src_key_padding_mask=~batch['mask'])
        return self.out(self.drop(h))

    def loss(self, batch):
        return self.crf.nll(self.emissions(batch), batch['y'], batch['mask'])

    @torch.no_grad()
    def decode(self, batch, omegas=None, mode='free'):
        """A boundary set for every word. omegas: the segmentations of Omega(w) for every word."""
        em = self.emissions(batch)
        paths, _ = self.crf.viterbi(em, batch['mask'])
        out = []
        lengths = batch['mask'].sum(1).tolist()
        A, st, en = (x.detach().cpu().numpy() for x in self.crf.params())
        em_np = em.detach().cpu().numpy()
        for b, path in enumerate(paths):
            if mode == 'free' or omegas is None:
                out.append(tags_to_cuts(path))
                continue
            L = lengths[b]
            best, best_s = None, -math.inf
            for cuts in omegas[b]:
                tags = cuts_to_tags(L, cuts)
                sc = st[tags[0]] + em_np[b, 0, tags[0]] + en[tags[-1]]
                for t in range(1, L):
                    sc += A[tags[t - 1], tags[t]] + em_np[b, t, tags[t]]
                if sc > best_s:
                    best, best_s = cuts, sc
            out.append(best)
        return out


# ----------------------------------------------------------------------------- using a saved model
def load_model(path):
    """Saved FEMSeg-uz model (.pt) -> (model, character vocabulary)."""
    ck = torch.load(path, weights_only=False)
    vocab = CharVocab([])
    vocab.itos = ck['vocab']
    vocab.stoi = {c: i for i, c in enumerate(vocab.itos)}
    model = FEMSegUz(len(vocab.itos), channels=ck['channels'], use_cnn=ck['use_cnn'],
                     use_transformer=ck['use_transformer'])
    model.load_state_dict(ck['state'])
    model.eval()
    return model, vocab


def predict_cuts(model, words, vocab, cands, mode='hard', bs=512):
    """List of words -> boundary sets (in the input order). A batch holds words of one length only, so there is no
    padding and the result for a word does not depend on the other words of the batch."""
    model.eval()
    by_len = {}
    for i, w in enumerate(words):
        by_len.setdefault(len(w), []).append(i)
    out = [None] * len(words)
    for L in sorted(by_len):
        idx = by_len[L]
        for j in range(0, len(idx), bs):
            part = idx[j:j + bs]
            chunk = [words[i] for i in part]
            cl = [cands(w) for w in chunk]
            batch = collate([(encode_word(w, vocab, c), None) for w, c in zip(chunk, cl)])
            omegas = [omega_cuts(w, c) for w, c in zip(chunk, cl)] if mode != 'free' else None
            for i, r in zip(part, model.decode(batch, omegas, mode=mode)):
                out[i] = r
    return out
