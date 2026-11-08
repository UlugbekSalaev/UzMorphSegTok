"""The complete set of endings (CSE) of UzMorphAnalyser and the division of a word into morphemes with it.

Each CSE entry is a suffix chain that follows a stem, for example (i)mizdan = (i)miz + dan.
The table lists the morphemes of a chain in two columns (lexical_affixes, syntactical_affixes), but their order and
spelling are not always those of the chain (lik+i -> ligi, mi at the end and so on).
The morphemes are therefore aligned with the chain by dynamic programming (decompose_all).

Allomorph symbols of the table:
  G = g/k/q (after k and q of the stem), Y = a/y, (i), (a), (u) are dropped after a vowel,
  (S) gives s after a vowel, (n) is dropped after n; the choice for T, A, Q, K and (t) depends on the stem itself.
"""
import csv
import itertools
from collections import Counter

from .text import normalize, is_word

VOWELS = set('aeiou')
ALLOMORPHS = {'G': ['g', 'k', 'q'], 'K': ['g', 'k'], 'Y': ['a', 'y'], 'T': ['t', 'd'],
              'Q': ['g', "g'", 'k', 'q'], 'S': ['s', 'y'], 'A': ['a', 'i']}


# ----------------------------------------------------------------------------- helpers
def ends_with_vowel(s):
    return s.endswith("o'") or (len(s) > 0 and s[-1] in VOWELS)


def starts_with_vowel(s):
    return s.startswith("o'") or (len(s) > 0 and s[0] in VOWELS)


def ending_parts(abstract):
    """'(i)mizdan' -> ['(i)', 'm', 'i', 'z', 'd', 'a', 'n']"""
    out, i = [], 0
    while i < len(abstract):
        if abstract[i] == '(':
            j = abstract.index(')', i)
            out.append(abstract[i:j + 1])
            i = j + 1
        else:
            out.append(abstract[i])
            i += 1
    return out


# ----------------------------------------------------------------------------- reading the table
def read_rows(path):
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------------- division into morphemes
def token_variants(tok):
    """Forms that a listed morpheme can take inside a chain: a bracketed part is kept, opened or dropped; a final k/q
    becomes g/g' before a vowel; a vowel-initial morpheme can be written with that vowel in brackets (imiz -> (i)miz)."""
    opts = [[p, p[1:-1], ''] if p.startswith('(') else [p] for p in ending_parts(tok)]
    out = {''.join(c).lower() for c in itertools.product(*opts)} - {''}
    out |= {s[:-1] + 'g' for s in out if s.endswith('k')} | {s[:-1] + "g'" for s in out if s.endswith('q')}
    out |= {f'({s[0]}){s[1:]}' for s in out if len(s) > 1 and s[0] in 'iau'}
    return out


def align(abstract, tokens):
    """Align a chain with its listed morphemes (in any order).
    Cost: 2 for an unmatched run, 1 for an unused morpheme. Result: (cost, [(piece, kind)])."""
    a = abstract.lower()
    n, m = len(a), len(tokens)
    variants = [token_variants(t) for t in tokens]
    best = {(0, 0): (0, [])}
    for i in range(n + 1):
        for mask in range(1 << m):
            if (i, mask) not in best:
                continue
            cost, seg = best[(i, mask)]

            def update(key, c, s):
                old = best.get(key)
                if old is None or c < old[0] or (c == old[0] and len(s) > len(old[1])):
                    best[key] = (c, s)

            for j in range(m):
                if not mask >> j & 1:
                    for v in variants[j]:
                        if a.startswith(v, i):
                            update((i + len(v), mask | 1 << j), cost, seg + [(abstract[i:i + len(v)], 'T')])
            if i < n:
                k = a.index(')', i) + 1 if a[i] == '(' else i + 1
                if seg and seg[-1][1] == 'G':
                    update((k, mask), cost, seg[:-1] + [(seg[-1][0] + abstract[i:k], 'G')])
                else:
                    update((k, mask), cost + 2, seg + [(abstract[i:k], 'G')])
    finals = [(c + m - bin(mask).count('1'), s) for (i, mask), (c, s) in best.items() if i == n]
    return min(finals, key=lambda x: (x[0], -len(x[1])))


def split_with_inventory(piece, inventory):
    """Split an unmatched run into the fewest known morphemes (it stays whole if that is not possible)."""
    a = piece.lower()
    n = len(a)
    best = {0: []}
    for i in range(n):
        if i not in best:
            continue
        for v in inventory:
            if a.startswith(v, i):
                j = i + len(v)
                cand = best[i] + [piece[i:j]]
                if j not in best or len(cand) < len(best[j]):
                    best[j] = cand
    return best.get(n, [piece])


def decompose_all(rows):
    """The morphemes of the chain of every entry (in abstract form)."""
    raw_tokens = {}
    freq = Counter()
    for r in rows:
        toks = (r['lexical_affixes'] + ' ' + r['syntactical_affixes']).replace(',', ' ').split()
        raw_tokens[r['id']] = toks
        freq.update(set(toks))
    inventory = {v for t, c in freq.items() if c >= 3 for v in token_variants(t)}
    split_inventory = {v for t, c in freq.items() if c >= 10 for v in token_variants(t)}
    result = {}
    for r in rows:
        cost, seg = align(r['affix'], raw_tokens[r['id']])
        pieces = []
        for piece, kind in seg:
            if kind == 'G':
                pieces += [(x, 'G') for x in split_with_inventory(piece, split_inventory)]
            else:
                pieces.append((piece, 'T'))
        # an unmatched remainder of one part is joined to a neighbour: (S)+i -> (S)i, m+i -> mi
        k = 0
        while k < len(pieces):
            piece, kind = pieces[k]
            if kind == 'G' and len(ending_parts(piece)) == 1 and len(pieces) > 1:
                if k > 0 and (pieces[k - 1][0] + piece).lower() in inventory:
                    pieces[k - 1:k + 1] = [(pieces[k - 1][0] + piece, 'T')]
                    continue
                if k + 1 < len(pieces) and (piece + pieces[k + 1][0]).lower() in inventory:
                    pieces[k:k + 2] = [(piece + pieces[k + 1][0], 'T')]
                    continue
            k += 1
        morphs = [p for p, _ in pieces]
        result[r['id']] = morphs
    return harmonize(result)


def harmonize(decomp):
    """The same suffix is sometimes written as one morpheme and sometimes as two (laring / lar+ing).
    If most entries split it, it is split in the others as well."""
    single = Counter(m.lower() for ms in decomp.values() for m in ms)
    pair = Counter()
    for ms in decomp.values():
        for a, b in zip(ms, ms[1:]):
            pair[(a.lower() + b.lower(), a, b)] += 1
    splits = {}
    for (joined, a, b), c in pair.items():
        if c > single.get(joined, 0) and single.get(joined, 0) > 0:
            if joined not in splits or c > splits[joined][2]:
                splits[joined] = (a, b, c)
    out = {}
    for eid, ms in decomp.items():
        new = []
        for m in ms:
            key = m.lower()
            if key in splits:
                a, b, _ = splits[key]
                new += [m[:len(a)], m[len(a):]]
            else:
                new.append(m)
        out[eid] = new
    return out


# ----------------------------------------------------------------------------- allomorphs
def realise_parts(parts, prev):
    """All possible spellings of one morpheme after the string prev."""
    outs = ['']
    for p in parts:
        new = []
        for out in outs:
            cur = prev + out
            if p.startswith('('):
                inner = p[1:-1]
                if inner in ('i', 'a', 'u'):
                    opts = [''] if ends_with_vowel(cur) else [inner]
                elif inner == 'S':
                    opts = ['s'] if ends_with_vowel(cur) else ['']
                elif inner == 'n':
                    opts = [''] if cur.endswith('n') else ['n']
                elif inner in ALLOMORPHS:
                    opts = ALLOMORPHS[inner] + ['']
                else:
                    opts = [normalize(inner), '']
            elif p == 'G':
                opts = ['k'] if cur.endswith('k') else ['q'] if cur.endswith('q') else ['g']
            elif p == 'Y':
                opts = ['y'] if ends_with_vowel(cur) else ['a']
            elif p == 'S':
                opts = ['s'] if ends_with_vowel(cur) else ['']
            elif p in ALLOMORPHS:
                opts = ALLOMORPHS[p]
            else:
                opts = [normalize(p)]
            new += [out + o for o in opts]
        outs = new
    return outs


def realise(morphs, stem):
    """Realizations of a CSE chain after a stem: [(spelling, [spellings of the morphemes])]."""
    results = [('', [])]
    for m in morphs:
        new = []
        for s, ms in results:
            for o in realise_parts(ending_parts(m), stem + s):
                if o:
                    new.append((s + o, ms + [o]))
        results = new
    return [(s, ms) for s, ms in results if s]


def surface_forms(morphs):
    """All spellings without context (for the index): stems ending in a vowel, k, q, n or another consonant."""
    out = set()
    for stem in ('kitob', 'ola', 'yurak', 'qishloq', 'xon'):
        for s, _ in realise(morphs, stem):
            out.add(s)
    return out


# ----------------------------------------------------------------------------- table and lexicon
class CSE:
    """CSE table: entries, their morphemes and an index by spelling."""

    def __init__(self, rows):
        decomp = decompose_all(rows)
        self.entries = []
        for r in rows:
            morphs = decomp[r['id']]
            self.entries.append({'id': int(r['id']), 'abstract': r['affix'], 'pos': r['pos'], 'morphs': morphs,
                                 'row': r, 'sample_roots': [normalize(x) for x in r['root'].split('/')
                                                            if is_word(normalize(x))]})
        self.index = {}
        for i, e in enumerate(self.entries):
            for s in surface_forms(e['morphs']):
                self.index.setdefault(s, []).append(i)
        self.max_len = max(len(s) for s in self.index)
        self.morph_inventory = Counter(m for e in self.entries for m in e['morphs'])

    def analyses(self, stem_surface, ending):
        """Entries that give `ending` after the written stem, with their morphemes."""
        out = []
        for i in self.index.get(ending, ()):
            for s, ms in realise(self.entries[i]['morphs'], stem_surface):
                if s == ending:
                    out.append((i, tuple(ms)))
        return out


def load_lexicon(path):
    """Stem lexicon: stem -> parts of speech."""
    lexicon = {}
    with open(path, encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            w = normalize(row['stem'])
            if is_word(w):
                lexicon.setdefault(w, set()).add(row['pos'].strip())
    return lexicon
