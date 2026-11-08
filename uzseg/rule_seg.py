"""CSE-Seg: rule-based segmenters built on the CSE table and a stem lexicon.

For a word w, the candidate set Omega(w) holds candidates: a stem length k, a CSE entry and the morphemes of its chain
(w[k:] = m_1 ... m_r). The unsegmented word (k = |w|) is always a candidate.
"""
import functools
import math
from collections import Counter

from .cse import ends_with_vowel, starts_with_vowel

NOMINAL_LEX = {'NOUN', 'ADJ', 'PRN', 'ADV', 'NUM', 'MOD', 'PPN'}
CLOSED = {'CNJ', 'PRT', 'ADP', 'INTJ', 'IMIT', 'AUX', 'MOD'}


def letters(s):
    out, i = [], 0
    while i < len(s):
        if s[i:i + 2] in ("o'", "g'", 'sh', 'ch', 'ng'):
            out.append(s[i:i + 2])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return out


def is_vowel_letter(u):
    return u == "o'" or u in 'aeiou'


class Candidates:
    """Builds Omega(w)."""

    def __init__(self, cse, lexicon, min_stem=2):
        self.cse, self.lexicon, self.min_stem = cse, lexicon, min_stem

    def lexical_stems(self, stem, ending):
        """Lexicon stems that the written stem can represent (k->g, q->g' and a dropped vowel are restored)."""
        out = []
        if stem in self.lexicon:
            out.append(stem)
        if starts_with_vowel(ending):
            if stem.endswith("g'") and stem[:-2] + 'q' in self.lexicon:
                out.append(stem[:-2] + 'q')               # o'rtog'i -> o'rtoq
            elif stem.endswith('g') and not stem.endswith('ng') and stem[:-1] + 'k' in self.lexicon:
                out.append(stem[:-1] + 'k')               # yuragi -> yurak
            u = letters(stem)
            if len(u) >= 3 and not is_vowel_letter(u[-1]) and not is_vowel_letter(u[-2]):
                for v in ('i', 'u', 'a', 'o'):            # shahri -> shahar, burni -> burun
                    full = ''.join(u[:-1]) + v + u[-1]
                    if full in self.lexicon:
                        out.append(full)
        return out

    @functools.lru_cache(maxsize=1_000_000)
    def __call__(self, w):
        cands = [{'k': len(w), 'entry': None, 'morphs': (), 'lex': [w] if w in self.lexicon else []}]
        for k in range(1, len(w)):
            stem, ending = w[:k], w[k:]
            if len(ending) > self.cse.max_len or ending.startswith("'"):
                continue
            if ends_with_vowel(stem) and starts_with_vowel(ending):
                continue
            lex = self.lexical_stems(stem, ending)
            if k < self.min_stem and not lex:
                continue
            seen = set()
            for ei, morphs in self.cse.analyses(stem, ending):
                if morphs in seen:
                    continue
                seen.add(morphs)
                cands.append({'k': k, 'entry': ei, 'morphs': morphs, 'lex': lex})
        return cands

    def pos_ok(self, cand):
        """Does the part of speech of the CSE entry agree with the lexicon stem (a verbal entry needs a verbal stem)?"""
        if cand['entry'] is None:
            return True
        pos = self.cse.entries[cand['entry']]['pos']
        tags = set().union(*(self.lexicon.get(r, set()) for r in cand['lex'])) if cand['lex'] else set()
        return ('VERB' in tags) if pos == 'VERB' else bool(tags & NOMINAL_LEX)


def segmentation(w, cand):
    """Candidate -> list of morphemes (the first one is the stem)."""
    return [w[:cand['k']]] + list(cand['morphs'])


def paradigm_stats(cands, word_freq, min_freq=2, max_len=30):
    """Paradigm statistics over the corpus word types.
    N(r): number of word types that have a candidate with stem r (the paradigm of the stem);
    F(e): frequency of an ending, shared equally among the candidates of a word ('' is the empty ending)."""
    stem_types, end_freq = Counter(), Counter()
    for w, f in word_freq.items():
        if f < min_freq or len(w) > max_len:
            continue
        cs = cands(w)
        stems = set()
        for c in cs:
            if c['k'] == len(w):
                stems.add(w)
                end_freq[''] += 1 / len(cs)
            else:
                stems.update(c['lex'] or [w[:c['k']]])
                end_freq[w[c['k']:]] += 1 / len(cs)
        for s in stems:
            stem_types[s] += 1
    return {'stem_types': dict(stem_types), 'end_freq': dict(end_freq)}


class CSESeg:
    """Rule-based segmenter.

    mode='longest' (CSE-Seg): the verified candidate with the longest ending; if there is none, the candidate with the
    longest ending whose stem has at least `oov_min_stem` letters.

    mode='paradigm' (CSE-Seg+P): candidates are scored by
        score(c) = log(1 + N(r_c)) + beta * log P(e_c) + gamma * [r_c is in the lexicon],
    where r_c is the stem of the candidate, e_c its ending and P(e) = (F(e) + 0.5) / (sum F + 0.5 |E|).
    If verified candidates exist, only they and the word itself are compared.
    Closed-class lexicon words (conjunctions, particles, postpositions and so on) are never divided."""

    def __init__(self, cands, mode='paradigm', stats=None, beta=0.1, gamma=2.0, oov_min_stem=2):
        self.cands, self.mode, self.stats = cands, mode, stats
        self.beta, self.gamma, self.oov_min_stem = beta, gamma, oov_min_stem
        if stats is not None:
            self.total = sum(stats['end_freq'].values())

    def score(self, w, c):
        st = self.stats
        if c['k'] == len(w):
            stem, ending = w, ''
            verified = w in self.cands.lexicon
        else:
            stem, ending = (c['lex'] or [w[:c['k']]])[0], w[c['k']:]
            verified = bool(c['lex'])
        p_end = (st['end_freq'].get(ending, 0.0) + 0.5) / (self.total + 0.5 * len(st['end_freq']))
        return math.log1p(st['stem_types'].get(stem, 0)) + self.beta * math.log(p_end) + self.gamma * verified

    @functools.lru_cache(maxsize=1_000_000)
    def segment(self, w):
        if self.mode == 'paradigm':
            return self.segment_paradigm(w)
        return self.segment_longest(w)

    def segment_paradigm(self, w):
        cs = self.cands(w)
        lex = self.cands.lexicon
        if w in lex and lex[w] & CLOSED:
            return (w,)
        verified = [c for c in cs[1:] if c['lex'] and self.cands.pos_ok(c)]
        pool = verified + [cs[0]] if verified else [cs[0]] + [c for c in cs[1:] if c['k'] >= self.oov_min_stem]
        best = max(pool, key=lambda c: (self.score(w, c), -c['k']))
        return tuple(segmentation(w, best))

    def segment_longest(self, w):
        cs = self.cands(w)
        lex = self.cands.lexicon
        if w in lex and lex[w] & CLOSED:
            return (w,)
        verified = [c for c in cs[1:] if c['lex'] and self.cands.pos_ok(c)]
        if verified:
            best = min(verified, key=lambda c: (c['k'], -len(c['morphs'])))
            return tuple(segmentation(w, best))
        if w in lex:
            return (w,)
        oov = [c for c in cs[1:] if c['k'] >= self.oov_min_stem]
        if oov:
            best = min(oov, key=lambda c: (c['k'], -len(c['morphs'])))
            return tuple(segmentation(w, best))
        return (w,)
