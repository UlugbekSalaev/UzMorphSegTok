"""Step 1. Prepare the data.

  * apertium-uzb lexicon + UzMorphAnalyser stem lists -> stem lexicon data/cse/stems.csv;
  * CSE table (snapshot of affixes.csv) -> data/cse/cse_morphs.tsv (morpheme decomposition);
  * UD_Uzbek-UzUDT r2.18 -> reference segmentation data/reference/*.jsonl;
  * Uzbek News Dataset v0.0.1 -> filtered documents, fixed splits (data/splits/*.txt),
    tokenizer corpus data/corpus/{train,test}.txt and classification data data/extrinsic/uztc_{train,dev,test}.jsonl.
Every filter is counted in results/corpus_stats.json and results/reference_stats.json.
"""
import hashlib
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from uzseg.cse import read_rows, CSE
from uzseg.reference import build_reference
from uzseg.text import (CYRILLIC, clean_sentence, has_foreign_letter, is_word, normalize, read_conllu,
                        split_sentences, word_spans)

# folder names of the news dataset and their English names (Zenodo record of the dataset)
CATEGORIES = {'Avto': 'Auto', 'Ayollar': 'Women', 'Dunyo': 'World', 'Foto': 'Photo', 'Iqtisodiyot': 'Economics',
              'Jamiyat': 'Society', 'Jinoyat': 'Crime', 'Madaniyat': 'Culture', "O'zbekiston": 'Local',
              'Pazandachilik': 'Culinary', 'Qonunchilik': 'Law', 'Salomatlik': 'Health', 'Siyosat': 'Politics',
              'Sport': 'Sport', 'Texnologiya': 'Tech'}


# continuation classes of apertium-uzb -> parts of speech of the lexicon (closed classes are never divided)
APERTIUM_POS = [('NP-', 'PPN'), ('N1', 'NOUN'), ('N2', 'NOUN'), ('TIT', 'NOUN'), ('A1', 'ADJ'), ('A2', 'ADJ'),
                ('A3', 'ADJ'), ('V-', 'VERB'), ('VERB-', 'VERB'), ('ADV', 'ADV'), ('NUM', 'NUM'), ('PRON', 'PRN'),
                ('DET', 'PRN'), ('POST', 'ADP'), ('PREP', 'ADP'), ('CC', 'CNJ'), ('CS', 'CNJ'), ('CA', 'CNJ'),
                ('IJ', 'INTJ'), ('IDEO', 'IMIT'), ('QST', 'PRT'), ('PAREN', 'MOD')]
APERTIUM_SKIP = ('MIST', 'N1-REMOVE', 'ABBR', '#')       # misspellings, removed entries, abbreviations


def check_files(folder, expected):
    for name, sha in expected.items():
        if hashlib.sha256(open(os.path.join(folder, name), 'rb').read()).hexdigest() != sha:
            sys.exit(f'{folder}/{name} differs from the fixed version')


def prepare_lexicon():
    """Stem lexicon = one-word entries of the apertium-uzb lexicon + three stem lists of UzMorphAnalyser."""
    check_files(C.APERTIUM_DIR, C.APERTIUM_FILES)
    check_files(C.UZMA_DIR, C.UZMA_SHA256)
    rows, stats = set(), Counter()
    entry = re.compile(r'^([^:!;]+?):(\S+)\s+(\S+)\s*;')
    for line in open(os.path.join(C.APERTIUM_DIR, 'apertium-uzb.uzb.lexc'), encoding='utf-8'):
        s = line.strip()
        if not s or s.startswith(('!', 'LEXICON')):
            continue
        m = entry.match(s)
        if not m:
            continue
        upper, lower, cont = m.groups()
        if '%<' in upper:                       # fully inflected forms of the closed lexicons
            continue
        stats['entries'] += 1
        pos = None if cont in APERTIUM_SKIP else next((p for pre, p in APERTIUM_POS if cont.startswith(pre)), None)
        w = normalize(lower.replace('%', ''))
        if pos is None:
            stats['skipped_class'] += 1
        elif not is_word(w):
            stats['skipped_multiword'] += 1
        else:
            rows.add((w, pos, 'apertium'))
    for name, default in (('non_affixed_stems.csv', 'PRT'), ('number_stems.csv', 'NUM'),
                          ('exception_stems.csv', 'NOUN')):
        lines = open(os.path.join(C.UZMA_DIR, name), encoding='utf-8').read().splitlines()
        header = lines[0].split(',')
        for line in lines[1:] if header[0] == 'stem' else lines:
            parts = line.split(',')
            w = normalize(parts[0])
            pos = default
            if name == 'exception_stems.csv' and len(parts) > 1 and parts[1].strip():
                pos = {'MOUN': 'NOUN', 'PRON': 'PRN'}.get(parts[1].strip(), parts[1].strip())
            if is_word(w):
                rows.add((w, pos, 'uzmorphanalyser'))
    with open(C.LEXICON, 'w', encoding='utf-8', newline='') as f:
        f.write('stem,pos,source\n')
        for w, pos, src in sorted(rows):
            f.write(f'{w},{pos},{src}\n')
    stems = {w for w, _, _ in rows}
    out = {**stats, 'stems': len(stems), 'rows': len(rows),
           'from_uzmorphanalyser_only': len(stems - {w for w, _, s in rows if s == 'apertium'}),
           'pos': dict(Counter(p for _, p, _ in rows))}
    json.dump(out, open(os.path.join(C.RESULTS, 'lexicon_stats.json'), 'w'), indent=2)
    print('lexicon', out)


def prepare_cse():
    check_files(C.UZMA_DIR, C.UZMA_SHA256)
    cse = CSE(read_rows(C.CSE_CSV))
    with open(os.path.join(C.CSE_DIR, 'cse_morphs.tsv'), 'w', encoding='utf-8') as f:
        f.write('id\tpos\tcse\tmorphs\n')
        for e in cse.entries:
            f.write(f"{e['id']}\t{e['pos']}\t{e['abstract']}\t{' + '.join(e['morphs'])}\n")
    print(f'CSE: {len(cse.entries)} entries, {len(cse.morph_inventory)} morpheme types')
    return cse


def prepare_reference(cse):
    stats_all = {}
    for name, fname in C.UD_FILES.items():
        ref, stats = build_reference(read_conllu(os.path.join(C.UD_DIR, fname)), cse, name)
        with open(os.path.join(C.REF_DIR, name + '.jsonl'), 'w', encoding='utf-8') as f:
            for s in ref:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        words = [w for s in ref for w in s['words']]
        stats_all[name] = {'sentences': len(ref), 'tokens': sum(len(s['forms']) for s in ref), 'words': len(words),
                           'inflected': sum(len(w['seg']) > 1 for w in words),
                           'positions': sum(len(w['w']) - 1 for w in words),
                           'boundaries': sum(len(w['seg']) - 1 for w in words),
                           'morphs_per_word': sum(len(w['seg']) for w in words) / len(words), **dict(stats)}
        print(name, stats_all[name])
    json.dump(stats_all, open(os.path.join(C.RESULTS, 'reference_stats.json'), 'w'), indent=2)
    return stats_all


def read_news():
    """All documents in a fixed order: (document id, folder name, text)."""
    with zipfile.ZipFile(C.NEWS_ZIP) as z:
        names = sorted(n for n in z.namelist() if n.endswith('.txt'))
        for name in names:
            _, folder, fname = name.split('/')
            folder = normalize(folder).capitalize()
            yield f'{folder}/{fname[:-4]}', folder, z.read(name).decode('utf-8', 'replace')


def filter_news():
    """Remove documents with Cyrillic letters, very short documents and duplicates.
    A text that occurs under two topic labels is removed completely, because its label is not defined."""
    stats, docs, groups = Counter(), [], defaultdict(list)
    per_cat = {k: Counter() for k in CATEGORIES}
    for doc_id, folder, raw in read_news():
        stats['all'] += 1
        cat = per_cat[folder]
        cat['all'] += 1
        if CYRILLIC.search(raw):
            stats['cyrillic'] += 1
            continue
        if len(word_spans(clean_sentence(raw))) < C.NEWS_MIN_WORDS:
            stats['short'] += 1
            continue
        key = hashlib.md5(normalize(clean_sentence(raw)).encode('utf-8')).hexdigest()
        groups[key].append(len(docs))
        docs.append((doc_id, folder, raw))
    drop = set()
    for idx in groups.values():
        if len(idx) > 1:
            labels = {docs[i][1] for i in idx}
            drop.update(idx if len(labels) > 1 else idx[1:])
            stats['duplicates' if len(labels) == 1 else 'label_conflict'] += len(idx) - 1 if len(labels) == 1 else len(idx)
    kept = [d for i, d in enumerate(docs) if i not in drop]
    stats['kept'] = len(kept)
    for doc_id, folder, raw in kept:
        per_cat[folder]['kept'] += 1
    return kept, stats, {k: dict(v) for k, v in per_cat.items()}


def title_of(raw):
    return normalize(clean_sentence(next(x for x in raw.splitlines() if x.strip())))


def classification_sample(kept, per_cat, rng):
    """Balanced sample of the categories with more than CLS_MIN_DOCS documents. Every title occurs once, so the same
    story cannot appear in two parts of the sample."""
    classes = sorted(k for k, v in per_cat.items() if v['kept'] > C.CLS_MIN_DOCS)
    by_cat = defaultdict(list)
    for d in kept:
        by_cat[d[1]].append(d)
    used_titles, out = set(), {'train': [], 'dev': [], 'test': []}
    selected = set()
    for cat in classes:
        pool = by_cat[cat]
        rng.shuffle(pool)
        sample = []
        for d in pool:
            t = title_of(d[2])
            if t in used_titles:
                continue
            used_titles.add(t)
            sample.append(d)
            if len(sample) == C.CLS_DRAWN_TRAIN + C.CLS_DEV + C.CLS_TEST:
                break
        selected.update(d[0] for d in sample)
        out['train'] += sample[:C.CLS_DRAWN_TRAIN]
        out['dev'] += sample[C.CLS_DRAWN_TRAIN:C.CLS_DRAWN_TRAIN + C.CLS_DEV]
        out['test'] += sample[C.CLS_DRAWN_TRAIN + C.CLS_DEV:]
    return out, selected, classes


def split_train(docs):
    """The first CLS_TRAIN documents of every category (in the shuffled order) form the training part of the main
    experiment, the others the reserve used only in the check with 5,000 training documents per category."""
    used, reserve, seen = [], [], Counter()
    for d in docs:
        seen[d[1]] += 1
        (used if seen[d[1]] <= C.CLS_TRAIN else reserve).append(d)
    return used, reserve


def write_ids(name, ids):
    with open(os.path.join(C.SPLIT_DIR, name + '.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(ids) + '\n')


def good_sentences(raw, stats):
    """Sentences of a document with SENT_MIN_WORDS to SENT_MAX_WORDS words and only basic Latin letters."""
    for s in split_sentences(raw):
        stats['sentences_all'] += 1
        if has_foreign_letter(s):
            stats['sentences_foreign_letter'] += 1
            continue
        n = len(word_spans(s))
        if n < C.SENT_MIN_WORDS or n > C.SENT_MAX_WORDS:
            stats['sentences_length'] += 1
            continue
        yield s


def prepare_news():
    rng = random.Random(C.SEED)
    kept, stats, per_cat = filter_news()
    print(dict(stats), flush=True)
    cls, selected, classes = classification_sample(kept, per_cat, rng)
    stats_cls = {}
    for part, docs in list(cls.items()):
        rng.shuffle(docs)
        parts = dict(zip(('train', 'reserve'), split_train(docs))) if part == 'train' else {part: docs}
        for name, ds in parts.items():
            write_ids('news_classification_' + name, [d[0] for d in ds])
            with open(os.path.join(C.EXT_DIR, f'uztc_{name}.jsonl'), 'w', encoding='utf-8') as f:
                for doc_id, folder, raw in ds:
                    text = ' '.join(clean_sentence(raw).split()[:C.CLS_WORDS])
                    f.write(json.dumps({'id': doc_id, 'text': text, 'label': CATEGORIES[folder]},
                                       ensure_ascii=False) + '\n')
            stats_cls[name] = dict(Counter(CATEGORIES[d[1]] for d in ds))
    json.dump(stats_cls, open(os.path.join(C.RESULTS, 'classification_stats.json'), 'w', encoding='utf-8'), indent=2)
    rest = [d for d in kept if d[0] not in selected]
    rng.shuffle(rest)
    ud_texts = {normalize(''.join(t['form'] for t in s['tokens']))
                for fname in C.UD_FILES.values() for s in read_conllu(os.path.join(C.UD_DIR, fname))}

    def key_of(s):
        return normalize(s).replace(' ', '')

    # held-out test sentences of the tokenizer corpus come from their own documents
    test, seen, sent_stats, i = [], set(), Counter(), 0
    test_ids = []
    while len(test) < C.INTRINSIC_TEST_SIZE:
        doc_id, _, raw = rest[i]
        i += 1
        test_ids.append(doc_id)
        for s in good_sentences(raw, sent_stats):
            key = key_of(s)
            if key in seen or key in ud_texts:
                continue
            seen.add(key)
            test.append(s)
    test = test[:C.INTRINSIC_TEST_SIZE]
    train_docs = rest[i:i + C.CORPUS_DOCS]
    write_ids('news_tokenizer_test', test_ids)
    write_ids('news_tokenizer_train', [d[0] for d in train_docs])
    train, dup = [], 0
    sent_stats = Counter()
    for doc_id, _, raw in train_docs:
        for s in good_sentences(raw, sent_stats):
            key = key_of(s)
            if key in seen or key in ud_texts:
                dup += 1
                continue
            seen.add(key)
            train.append(s)
    open(os.path.join(C.CORPUS_DIR, 'train.txt'), 'w', encoding='utf-8').write('\n'.join(train) + '\n')
    open(os.path.join(C.CORPUS_DIR, 'test.txt'), 'w', encoding='utf-8').write('\n'.join(test) + '\n')
    out = {**stats, 'per_category': per_cat, 'classes': classes, 'test_docs': len(test_ids),
           'train_docs': len(train_docs), 'unused_docs': len(rest) - i - len(train_docs),
           'train': len(train), 'test': len(test), 'duplicate_sentences': dup, **sent_stats,
           'train_words': sum(len(word_spans(s)) for s in train), 'test_words': sum(len(word_spans(s)) for s in test)}
    print({k: v for k, v in out.items() if k != 'per_category'})
    json.dump(out, open(os.path.join(C.RESULTS, 'corpus_stats.json'), 'w', encoding='utf-8'), indent=2)


if __name__ == '__main__':
    prepare_lexicon()
    cse = prepare_cse()
    prepare_reference(cse)
    prepare_news()
