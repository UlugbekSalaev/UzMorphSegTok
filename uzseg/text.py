"""Text normalization, word and sentence splitting, and CoNLL-U reading.

Internally o', g' and the tutuq sign are written with one plain apostrophe (').
External (multilingual) tokenizers receive the official spelling: U+02BB after o and g, U+02BC elsewhere.
The replacement is one character for one character, so character positions do not change.
"""
import re
import unicodedata

APOSTROPHES = '`‘’ʻʼ´ʹ\''
WORD_RE = re.compile(r"[A-Za-z']+")
CYRILLIC = re.compile('[Ѐ-ӿ]')


def unify_apostrophes(text):
    for ch in APOSTROPHES:
        text = text.replace(ch, "'")
    for ch in '­​‌‍﻿️':       # soft hyphen and zero-width characters
        text = text.replace(ch, '')
    return text.replace('\xa0', ' ')


def normalize(word):
    """One form for comparing words: lower case, every apostrophe variant written as '."""
    w = unify_apostrophes(word.strip().lower())
    return w.replace('ö', "o'").replace('ğ', "g'").replace('ş', 'sh').replace('ç', 'ch')


def is_word(w):
    """Only Latin letters and the apostrophe (not a number or punctuation)."""
    return bool(re.fullmatch(r"[a-z']+", w)) and not w.startswith("'") and not w.endswith("''")


def clean_sentence(s):
    """Bring a corpus sentence to the internal form (letter case is kept)."""
    s = unify_apostrophes(s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def to_official(s):
    """Internal form -> official spelling: U+02BB after o and g, U+02BC elsewhere (one to one)."""
    out = []
    for i, ch in enumerate(s):
        if ch == "'":
            prev = s[i - 1].lower() if i > 0 else ''
            out.append('ʻ' if prev in ('o', 'g') else 'ʼ')
        else:
            out.append(ch)
    return ''.join(out)


def word_spans(s):
    """Words of a sentence as (start, end) pairs. A word is a run of Latin letters and inner apostrophes."""
    spans = []
    for m in WORD_RE.finditer(s):
        a, b = m.start(), m.end()
        while a < b and s[a] == "'":          # an apostrophe used as a quotation mark
            a += 1
        while b > a and s[b - 1] == "'" and not (b - 2 >= a and s[b - 2].lower() in 'og'):
            b -= 1
        if b > a:
            spans.append((a, b))
    return spans


# A sentence ends at . ! ? or an ellipsis before a space and a capital letter, a digit or an opening quotation mark.
SENT_END = re.compile(r'[.!?…]+["”»\')]*\s+(?=["“«\'(]?[A-Z0-9])')
INITIAL = re.compile(r"(?:^|[\s(\"“«.])(?:[A-Z]|Sh|Ch|Ng|O'|G')$")


def split_sentences(text):
    """Rule-based sentence splitter for a document (the title and each line are separate units).
    A full stop after an initial (A. Qodiriy, Sh. Rashidov) does not end a sentence."""
    out = []
    for line in text.splitlines():
        line = clean_sentence(line)
        if not line:
            continue
        last = 0
        for m in SENT_END.finditer(line):
            if INITIAL.search(line[last:m.start()]):
                continue
            out.append(line[last:m.end()].strip())
            last = m.end()
        if last < len(line):
            out.append(line[last:].strip())
    return out


def has_foreign_letter(s):
    """True if the text contains a letter outside the basic Latin alphabet (Arabic, accented Latin and others)."""
    return any(ord(ch) > 127 and unicodedata.category(ch).startswith('L') for ch in s)


def strip_infinitive(lemma):
    """Remove the infinitive suffix -moq/-mak from a verb lemma: ko'rmoq -> ko'r."""
    if lemma.endswith(('moq', 'mak')) and len(lemma) > 4:
        return lemma[:-3]
    return lemma


def read_conllu(path):
    """List of sentences; each token has form, lemma, upos and feats (a dictionary)."""
    sentences, tokens, text = [], [], None
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\r\n')
        if not line:
            if tokens:
                sentences.append({'text': text, 'tokens': tokens})
                tokens, text = [], None
            continue
        if line.startswith('#'):
            if line.startswith('# text'):
                text = line.split('=', 1)[1].strip()
            continue
        cols = line.split('\t')
        if len(cols) < 10 or '-' in cols[0] or '.' in cols[0]:
            continue
        feats = {}
        if cols[5] != '_':
            for kv in cols[5].split('|'):
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    feats[k] = v
        tokens.append({'form': unify_apostrophes(cols[1]), 'lemma': unify_apostrophes(cols[2]),
                       'upos': cols[3], 'feats': feats})
    if tokens:
        sentences.append({'text': text, 'tokens': tokens})
    return sentences
