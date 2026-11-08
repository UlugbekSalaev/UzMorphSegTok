"""Synthetic word forms (source S of FEMSeg-uz): a lexicon stem + a CSE chain, realized with the allomorph rules.

All lemmas of the reference treebank are removed from the stems, so the stems of test words do not occur here.
Chains with stem-dependent symbols (T, A, Q, K) and with voice suffixes are used only with the sample roots of the table.
"""
import random
import re

from .cse import realise, starts_with_vowel, ends_with_vowel, ALLOMORPHS, ending_parts
from .text import is_word

VOICE_ONSETS = ('(i)l', '(i)n', '(i)t', '(i)sh', 'il', 'iz', 'Gaz', 'Giz', 'gaz', 'giz', 'sat', 'dir', 'tir',
                '(t)', 'ish', 'Qaz', 'Ar', 'Yver')
VOICE_SUFFIXES = ('tir', 'dir', 'giz', 'kiz', 'qiz', 'gaz', 'kaz', 'qaz', 'sat', 'il', 'in', 'ish', 'iz', 'ar',
                  'ir', 'l', 'n', 't', 'sh')
ELISION = re.compile(r"^[^aeiou]*[aeiou][^aeiou]*[iu][lnrz]$")
NOMINAL = ('NOUN', 'ADJ')


def is_lexical(morphs):
    for m in morphs:
        for p in ending_parts(m):
            inner = p[1:-1] if p.startswith('(') else p
            if inner in ('T', 'A', 'Q', 'K', 't'):
                return True
    return False


def is_restricted(entry):
    """Entries used only with their sample roots."""
    if entry['pos'] not in ('NOUN', 'VERB'):
        return True
    if entry['row'].get('verb_voice1', '') in ('Reflexive', 'Passive', 'Causative', 'Reciprocal'):
        return True
    return entry['pos'] == 'VERB' and entry['abstract'].startswith(VOICE_ONSETS)


def junction(root, ending, pos):
    """0: plain attachment, 1: k/q changes to g/g' (yurak+i -> yuragi), -1: such a form does not exist."""
    if ends_with_vowel(root) and starts_with_vowel(ending):
        return -1
    if pos in NOMINAL and starts_with_vowel(ending):
        if root[-1] in 'kq' and len(root) > 3:
            return 1
        if ELISION.match(root.replace("o'", 'o').replace("g'", 'g')):
            return -1
    return 0


def root_pools(cse, lexicon, exclude):
    long_surfaces = {s for s in cse.index if len(s) >= 3}
    verbs = {w for w, p in lexicon.items() if 'VERB' in p}
    pools = {'NOUN': [], 'VERB': []}
    for w, pos_set in lexicon.items():
        if w in exclude or not is_word(w) or w.endswith(('moq', 'mak')):
            continue
        for p in pos_set:
            key = 'VERB' if p == 'VERB' else 'NOUN' if p in NOMINAL else None
            if key is None:
                continue
            lo, hi = (3, 12) if key == 'NOUN' else (2, 9)
            if not lo <= len(w) <= hi:
                continue
            if any(w[-k:] in long_surfaces for k in range(3, len(w) - 1)):
                continue
            if key == 'VERB' and any(w.endswith(sx) and w[:-len(sx)] in verbs and len(w) - len(sx) >= 2
                                     for sx in VOICE_SUFFIXES):
                continue
            pools[key].append(w)
    return {k: sorted(set(v)) for k, v in pools.items()}


def synthetic_words(cse, lexicon, exclude, n_noun=300, n_verb=120, n_bare=20000, seed=2026):
    """Result: [(word, morphemes, stem)], word forms without repetition."""
    rng = random.Random(seed)
    pools = root_pools(cse, lexicon, exclude)
    items = []
    for e in cse.entries:
        if is_lexical(e['morphs']):
            continue
        pos = 'VERB' if e['pos'] == 'VERB' else 'NOUN'
        roots = [r for r in e['sample_roots'] if r not in exclude]
        if not is_restricted(e):
            n = n_noun if pos == 'NOUN' else n_verb
            roots += rng.sample(pools[pos], min(n, len(pools[pos])))
        for r in roots:
            forms = realise(e['morphs'], r)
            if len(forms) != 1:
                continue
            ending, morphs = forms[0]
            j = junction(r, ending, pos)
            if j < 0:
                continue
            stem = r[:-1] + ('g' if r[-1] == 'k' else "g'") if j == 1 else r
            if j == 1:
                forms = realise(e['morphs'], stem)
                if len(forms) != 1 or forms[0][0] != ending:
                    continue
            items.append((stem + ending, (stem,) + tuple(morphs), r))
    all_roots = sorted(set(pools['NOUN']) | set(pools['VERB']))
    for r in rng.sample(all_roots, min(n_bare, len(all_roots))):
        items.append((r, (r,), r))
    seen, out = set(), []
    for w, seg, r in items:
        if w not in seen and is_word(w) and len(w) <= 30:
            seen.add(w)
            out.append((w, seg, r))
    rng.shuffle(out)
    return out
