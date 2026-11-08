"""Phonological features of Uzbek: letters (o', g', sh, ch, ng), sound classes and syllable structure.

Literary Uzbek has lost most of the vowel harmony of other Turkic languages, so the allomorph of a suffix depends mainly
on the last sound of the stem: vowel or consonant ((S)i, Y, (i)), or k and q (G = g/k/q). The model receives for
every character:
  phon: sound class (1 vowel, 2 voiceless consonant, 3 voiced consonant, 4 sonorant, 5 apostrophe or other);
  digr: position in a two-character letter (0 single, 1 first, 2 second character);
  syll: role in the syllable (1 onset, 2 nucleus (vowel), 3 coda).
"""
VOWELS = ('a', 'e', 'i', 'o', 'u', "o'")
VOICELESS = ('p', 't', 'k', 'q', 's', 'f', 'x', 'h', 'sh', 'ch', 'c')
VOICED = ('b', 'd', 'g', "g'", 'z', 'j', 'v')
SONORANT = ('m', 'n', 'ng', 'l', 'r', 'y', 'w')
DIGRAPHS = ("o'", "g'", 'sh', 'ch', 'ng')


def letter_units(w):
    """Divide a word into letters: [(letter, start)]."""
    out, i = [], 0
    while i < len(w):
        if w[i:i + 2] in DIGRAPHS:
            out.append((w[i:i + 2], i))
            i += 2
        else:
            out.append((w[i], i))
            i += 1
    return out


def phon_class(u):
    if u in VOWELS:
        return 1
    if u in VOICELESS:
        return 2
    if u in VOICED:
        return 3
    if u in SONORANT:
        return 4
    return 5


def syllable_roles(units):
    """Role of every letter in its syllable. A vowel is the nucleus; of the consonants between two vowels the last one
    begins the next syllable (VC.CV, VCC.CV); consonants at the start of a word are onsets and at its end codas."""
    n = len(units)
    vowel = [phon_class(u) == 1 for u, _ in units]
    roles = [0] * n
    idx = [i for i in range(n) if vowel[i]]
    if not idx:
        return [1] * n
    for i in idx:
        roles[i] = 2
    for i in range(idx[0]):
        roles[i] = 1
    for i in range(idx[-1] + 1, n):
        roles[i] = 3
    for a, b in zip(idx, idx[1:]):
        cons = list(range(a + 1, b))
        if not cons:
            continue
        for j in cons[:-1]:
            roles[j] = 3
        roles[cons[-1]] = 1
    return roles


def char_features(w):
    """A triple (phon, digr, syll) for every character."""
    units = letter_units(w)
    roles = syllable_roles(units)
    feats = [None] * len(w)
    for (u, start), r in zip(units, roles):
        pc = phon_class(u)
        if len(u) == 2:
            feats[start] = (pc, 1, r)
            feats[start + 1] = (pc, 2, r)
        else:
            feats[start] = (pc, 0, r)
    return feats


def syllabify(w):
    """Division into syllables (for examples): kitoblarimizda -> ki-tob-la-ri-miz-da."""
    units = letter_units(w)
    roles = syllable_roles(units)
    out, cur = [], ''
    for (u, _), r in zip(units, roles):
        if r == 1 and any(phon_class(x) == 1 for x in cur_units(cur)):
            out.append(cur)
            cur = ''
        cur += u
    out.append(cur)
    return '-'.join(out)


def cur_units(s):
    return [u for u, _ in letter_units(s)]
