"""Reference segmentation from a UD treebank (manually annotated lemmas and morphological features).

Given a word w and its lemma l:
  1) stem boundary: w = s + e, where s is a written form of l (identical, k->g, q->g', or with a dropped vowel);
  2) the ending e is divided into morphemes with the CSE table; if several divisions exist, the CSE entry that agrees
     best with the UD features is chosen.
Step 1 does not use the CSE table, so the stem boundary is an independent reference for every word whose lemma can be
aligned. A word enters the full reference set only if step 2 also succeeds.
"""
from collections import Counter

from .text import normalize, is_word, strip_infinitive

UD_TO_CSE = {
    ('Case', 'Gen'): ('cases', 'Genitive'), ('Case', 'Acc'): ('cases', 'Accusative'),
    ('Case', 'Dat'): ('cases', 'Dative'), ('Case', 'Loc'): ('cases', 'Locative'),
    ('Case', 'Abl'): ('cases', 'Ablative'), ('Number', 'Plur'): ('plural', '1'),
    ('Tense', 'Past'): ('tense', 'Past'), ('Tense', 'Pres'): ('tense', 'Present'),
    ('Tense', 'Fut'): ('tense', 'Future'), ('VerbForm', 'Part'): ('verb_func', 'Participle'),
    ('VerbForm', 'Conv'): ('verb_func', 'Adverbial'), ('VerbForm', 'Inf'): ('verb_func', 'Infinitive'),
    ('VerbForm', 'Vnoun'): ('verb_func', 'Infinitive'), ('Polarity', 'Neg'): ('negative', '1'),
    ('Voice', 'Pass'): ('verb_voice1', 'Passive'), ('Voice', 'Cau'): ('verb_voice1', 'Causative'),
    ('Mood', 'Imp'): ('impulsion', 'Imperative'), ('Mood', 'Cnd'): ('impulsion', 'Conditional'),
}


def agreement(row, feats):
    """(number of UD features confirmed by the CSE entry, number of UD features that have a CSE counterpart)."""
    good = total = 0
    for (k, v), (col, val) in UD_TO_CSE.items():
        if feats.get(k) == v:
            total += 1
            good += row.get(col, '') == val
    for ud, col in (('Person[psor]', 'possession'), ('Person', 'person')):
        person = feats.get(ud)
        if person:
            total += 1
            good += row.get(col, '')[:1] == person
    return good, total


def stem_cut(w, lemma):
    """Length of the written stem in the word (k), or None if the lemma cannot be aligned with the word."""
    if w.startswith(lemma):
        return len(lemma)
    if lemma.endswith('k') and w.startswith(lemma[:-1] + 'g'):
        return len(lemma)
    if lemma.endswith('q') and w.startswith(lemma[:-1] + "g'"):
        return len(lemma) + 1
    if len(lemma) >= 4 and lemma[-2] in 'aiuo' and lemma[-1] not in "aeiou'" and lemma[-3] not in "aeiou'":
        short = lemma[:-2] + lemma[-1]                    # shahar -> shahr, burun -> burn, o'g'il -> o'g'l
        if w.startswith(short):
            return len(short)
    return None


def word_lemma(token):
    """Normalized word and lemma of a UD token (verb lemmas without the infinitive suffix); None if not a word."""
    w = normalize(token['form'])
    if not is_word(w) or len(w) < 2:
        return None
    lemma = normalize(token['lemma'])
    if token['upos'] in ('VERB', 'AUX'):
        lemma = strip_infinitive(lemma)
    return w, lemma


def reference_segmentation(token, cse, stats=None):
    """(word, morphemes) for one UD token, or None if the token does not enter the reference set."""
    stats = stats if stats is not None else Counter()
    pair = word_lemma(token)
    if pair is None:
        stats['not_word'] += 1
        return None
    w, lemma = pair
    if w == lemma:
        stats['bare'] += 1
        return w, (w,)
    k = stem_cut(w, lemma)
    if k is None or k >= len(w) or k < 1:
        stats['lemma_mismatch'] += 1
        return None
    stem, ending = w[:k], w[k:]
    analyses = cse.analyses(stem, ending)
    if not analyses:
        stats['ending_not_in_cse'] += 1
        return None
    options = {}
    for ei, morphs in analyses:
        good, total = agreement(cse.entries[ei]['row'], token['feats'])
        if morphs not in options or good > options[morphs][0]:
            options[morphs] = (good, total)
    if len(options) > 1:
        stats['ambiguous'] += 1
    best, (good, total) = max(options.items(), key=lambda x: (x[1][0], -len(x[0])))
    stats['inflected'] += 1
    stats['features_with_counterpart'] += total
    stats['features_confirmed'] += good
    stats['words_with_features'] += total > 0
    return w, (stem,) + best


def stem_reference(token):
    """Stem boundary of a UD token from its lemma alone: (word, k) with k = len(word) for a bare form, or None."""
    pair = word_lemma(token)
    if pair is None:
        return None
    w, lemma = pair
    if w == lemma:
        return w, len(w)
    k = stem_cut(w, lemma)
    if k is None or k >= len(w) or k < 1:
        return None
    return w, k


def build_reference(sentences, cse, source):
    """Reference per sentence: 'words' have the full segmentation, 'stems' only the lemma-based stem boundary."""
    stats = Counter()
    out = []
    for i, s in enumerate(sentences):
        words, stems = [], []
        for j, t in enumerate(s['tokens']):
            r = reference_segmentation(t, cse, stats)
            if r is not None:
                words.append({'i': j, 'w': r[0], 'seg': list(r[1]), 'form': t['form'], 'upos': t['upos']})
            sr = stem_reference(t)
            if sr is not None:
                stems.append({'i': j, 'w': sr[0], 'k': sr[1], 'in_set': r is not None})
        out.append({'id': f'{source}:{i}', 'src': source, 'words': words, 'stems': stems,
                    'forms': [t['form'] for t in s['tokens']]})
    return out, stats
