"""Tokenizers: our BPE and Unigram models (HuggingFace Tokenizers) and the tokenizers of three multilingual models.

Pre-tokenization of our models:
    1) split at spaces, a marker U+2581 is attached to the start of a word (the SentencePiece convention);
    2) split at the morpheme boundary character U+2AF6, which is then removed;
    3) every digit becomes a separate token.
No piece of a model trained on a segmented corpus therefore crosses a morpheme boundary of that corpus.

A text is encoded in one of two modes:
    raw:       the text is encoded as it is (no segmenter is needed);
    segmented: U+2AF6 is first inserted at the boundaries found by a segmenter, so tokens cannot cross these boundaries.
For every token, its position (start, end) in the original text is returned.
"""
import os

from tokenizers import Tokenizer, models, trainers, pre_tokenizers

from .text import to_official, word_spans, normalize, is_word

DELIM = '⫶'          # morpheme boundary character


def pre_tokenizer():
    return pre_tokenizers.Sequence([
        pre_tokenizers.Metaspace(replacement='▁', prepend_scheme='always', split=True),
        pre_tokenizers.Split(DELIM, behavior='removed'),
        pre_tokenizers.Digits(individual_digits=True),
    ])


def corpus_lines(path):
    """The sentences of a training file without line ends (CR LF or LF), so that no piece contains a line end."""
    with open(path, encoding='utf-8') as f:
        for line in f:
            yield line.rstrip('\r\n')


def train_tokenizer(input_path, out_path, vocab_size, model_type):
    """Train a BPE or Unigram model on the sentences of a text file (one per line) and save it as a JSON file."""
    if model_type == 'bpe':
        tok = Tokenizer(models.BPE(unk_token='<unk>'))
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=['<unk>'], min_frequency=2,
                                      limit_alphabet=300, show_progress=False)
    else:
        tok = Tokenizer(models.Unigram())
        trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=['<unk>'], unk_token='<unk>',
                                          max_piece_length=16, shrinking_factor=0.75, show_progress=False)
    tok.pre_tokenizer = pre_tokenizer()
    tok.train_from_iterator(corpus_lines(input_path), trainer)
    tok.save(out_path)
    return out_path


def insert_boundaries(text, segmenter):
    """Insert U+2AF6 inside words at the boundaries found by the segmenter.
    Result: (new text, position of every new character in the original text, or -1)."""
    out, pos = [], []
    last = 0
    spans = word_spans(text)
    words = [normalize(text[a:b]) for a, b in spans]
    segmenter.prefetch([w for w in words if is_word(w)])
    for (a, b), w in zip(spans, words):
        for i in range(last, a):
            out.append(text[i])
            pos.append(i)
        cuts = segmenter(w) if is_word(w) and len(w) == b - a else set()
        for k in range(a, b):
            if k - a in cuts:
                out.append(DELIM)
                pos.append(-1)
            out.append(text[k])
            pos.append(k)
        last = b
    for i in range(last, len(text)):
        out.append(text[i])
        pos.append(i)
    return ''.join(out), pos


class LocalTokenizer:
    """Our tokenizer; with a segmenter it encodes in the segmented mode."""

    def __init__(self, name, path, segmenter=None):
        self.name, self.segmenter = name, segmenter
        self.tok = Tokenizer.from_file(path)
        self.vocab_size = self.tok.get_vocab_size()

    def encode(self, text):
        """[(token_id, start, end)]: positions are characters of the original text."""
        if self.segmenter is None:
            enc = self.tok.encode(text)
            return [(i, a, b) for i, (a, b) in zip(enc.ids, enc.offsets)]
        s, pos = insert_boundaries(text, self.segmenter)
        enc = self.tok.encode(s)
        out = []
        for i, (a, b) in zip(enc.ids, enc.offsets):
            orig = [pos[j] for j in range(a, b) if pos[j] >= 0]
            if orig:
                out.append((i, orig[0], orig[-1] + 1))
            else:
                k = next((pos[j] for j in range(a, len(pos)) if pos[j] >= 0), len(text))
                out.append((i, k, k))
        return out

    def ids(self, text):
        return [i for i, _, _ in self.encode(text)]


class HFTokenizer:
    """Tokenizer of a multilingual model. The text is given in the official spelling (one to one replacement)."""

    def __init__(self, name, hf_id, revision=None):
        from transformers import AutoTokenizer
        self.name, self.hf_id = name, hf_id
        self.tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
        self.vocab_size = len(self.tok)

    def encode(self, text):
        enc = self.tok(to_official(text), return_offsets_mapping=True, add_special_tokens=False)
        return [(i, a, b) for i, (a, b) in zip(enc['input_ids'], enc['offset_mapping'])]

    def ids(self, text):
        return [i for i, _, _ in self.encode(text)]


def word_cuts(encoded, spans):
    """For every word (a, b), the token boundaries inside it (positions counted from the start of the word)."""
    ends = sorted({b for _, a, b in encoded if b > a})
    out = []
    for a, b in spans:
        out.append({e - a for e in ends if a < e < b})
    return out


HF_TOKENIZERS = {          # model id and the commit of the HuggingFace repository
    'mT5': ('google/mt5-base', '2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f'),
    'XLM-R': ('FacebookAI/xlm-roberta-base', 'e73636d4f797dec63c3081bb6ed5c7b0bb3f2089'),
    'mBERT': ('google-bert/bert-base-multilingual-cased', '3f076fdb1ab68d5b2880cb87a0886f315b8146f8'),
}


def load_tokenizers(tok_dir, names):
    """Tokenizers by name. '<model>+seg' is the segmented mode: *_cse_* models use CSE-Seg+P and *_fem_* models
    use FEMSeg-uz."""
    from .segmenters import CSESegmenter, FEMSegmenter, candidates
    toks, segs, cands = {}, {}, None
    for name in names:
        if name in HF_TOKENIZERS:
            toks[name] = HFTokenizer(name, *HF_TOKENIZERS[name])
            continue
        base, seg = (name[:-4], True) if name.endswith('+seg') else (name, False)
        segmenter = None
        if seg:
            kind = 'fem' if '_fem_' in base else 'cse'
            if kind not in segs:
                cands = cands or candidates()
                segs[kind] = FEMSegmenter(cands=cands) if kind == 'fem' else CSESegmenter(cands=cands)
            segmenter = segs[kind]
        toks[name] = LocalTokenizer(name, os.path.join(tok_dir, base + '.json'), segmenter)
    return toks
