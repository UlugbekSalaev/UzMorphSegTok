"""Project settings: file paths and main hyperparameters (in one place)."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
RAW = os.path.join(DATA, 'raw_download')
CSE_DIR = os.path.join(DATA, 'cse')
UD_DIR = os.path.join(RAW, 'UD_Uzbek-UzUDT_r2.18')
CORPUS_DIR = os.path.join(DATA, 'corpus')
REF_DIR = os.path.join(DATA, 'reference')
EXT_DIR = os.path.join(DATA, 'extrinsic')
MODELS = os.path.join(ROOT, 'models')
TOK_DIR = os.path.join(MODELS, 'tokenizers')
SEG_DIR = os.path.join(MODELS, 'femseg')
RESULTS = os.path.join(ROOT, 'results')
FIGURES = os.path.join(RESULTS, 'figures')
TABLES = os.path.join(RESULTS, 'tables')
PAPER = os.path.join(ROOT, 'paper')

# source 1: UD_Uzbek-UzUDT, release 2.18 (git tag r2.18)
UD_TAG = 'r2.18'
UD_COMMIT = '170a1b303619b58273b4bc84fe45c241a039101f'
UD_BASE = f'https://raw.githubusercontent.com/UniversalDependencies/UD_Uzbek-UzUDT/{UD_COMMIT}/'
UD_URLS = {name: UD_BASE + name for name in ('uz_uzudt-ud-train.conllu', 'uz_uzudt-ud-test.conllu', 'LICENSE.txt',
                                             'README.md')}
UD_SHA256 = {'uz_uzudt-ud-train.conllu': 'c1fb9548cbba34e3dddd5ca2af7d5eb3e37f1e8a6e640e29061ff5b9e581aa12',
             'uz_uzudt-ud-test.conllu': '6a191e59b6cbaade250463109581e95350702eb85fff765f407eca00ed1937dc',
             'LICENSE.txt': '899b1804a12ebc090b96339614eede1b64b686721b650a71430b55b5235f7f79',
             'README.md': '8b908a10d956c08afcf985641557668e916966524e9497566b7cf0a8fd4eb399'}
UD_FILES = {'UzUDT_train': 'uz_uzudt-ud-train.conllu', 'UzUDT_test': 'uz_uzudt-ud-test.conllu'}
DEV_SET = 'UzUDT_train'                       # the official train part: 80% gold training data, 20% dev
TEST_SETS = ['UzUDT_test']                    # the official test part is used only for the final evaluation

# source 2: Uzbek News Dataset v0.0.1 (Kuriyozov et al., 2023)
NEWS_URL = 'https://zenodo.org/api/records/7677431/files/Uzbek_News_Dataset.zip/content'
NEWS_MD5 = 'fd50d6f0a0cd1bb24d44b1e1607b0320'
NEWS_ZIP = os.path.join(RAW, 'Uzbek_News_Dataset_v0.0.1', 'Uzbek_News_Dataset.zip')

# knowledge resource 1: the ending table and three stem lists of UzMorphAnalyser (Salaev, 2024), files of the PyPI
# package version 0.0.22, kept as a snapshot in data/cse/uzmorphanalyser_0.0.22
UZMA_DIR = os.path.join(CSE_DIR, 'uzmorphanalyser_0.0.22')
CSE_CSV = os.path.join(UZMA_DIR, 'affixes.csv')
UZMA_SHA256 = {'affixes.csv': 'de50fff9d98aa20f26d01f4a82c87f4a59c797893af89ee9c485ff653c429b62',
               'non_affixed_stems.csv': 'c69dc49c9770e4eca04af2e19fcfcbffaf1d80ecb9fb31e1b92f519889ed8787',
               'number_stems.csv': '070e41a0a32f31674a614782266d2db5b59aa5af50bd067d877eb7623dde42d2',
               'exception_stems.csv': '36a9c6a0613d3602a707f33ec072a253759ea6f22a8acc6c6bdb79d2b69478bf'}

# knowledge resource 2: the stem lexicon of the Apertium morphological transducer for Uzbek (GPL-3.0)
APERTIUM_COMMIT = '1bdee4a38c357226176d516ae533c43f8ecb045b'
APERTIUM_DIR = os.path.join(RAW, 'apertium-uzb_' + APERTIUM_COMMIT[:7])
APERTIUM_BASE = f'https://raw.githubusercontent.com/apertium/apertium-uzb/{APERTIUM_COMMIT}/'
APERTIUM_FILES = {'apertium-uzb.uzb.lexc': '18281d405c44437f5d1b2c67e71d8b31f3d44fc316c223454714c27d81631ee4',
                  'COPYING': '8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903',
                  'AUTHORS': '2b4d4aae3e001a718113fbf2100d87f7fca2aa6900449dd3889f59c59a4e3ab4'}
LEXICON = os.path.join(CSE_DIR, 'stems.csv')       # built by step 1 from the two resources above

# news documents: filters and sizes of the parts (see scripts/s01_prepare_data.py)
NEWS_MIN_WORDS = 30                           # shorter documents are removed
SENT_MIN_WORDS, SENT_MAX_WORDS = 4, 60        # sentences of the tokenizer corpus
CORPUS_DOCS = 100000                          # documents of the tokenizer training corpus
CLS_MIN_DOCS = 10000                          # topic categories with more documents are used for classification
CLS_DRAWN_TRAIN, CLS_DEV, CLS_TEST = 5000, 500, 1000   # documents drawn per category for training, dev and test
CLS_TRAIN = 2000                              # training documents per category of the main experiment; the other
                                              # 3000 drawn ones are used only in the check with 5000 documents
CLS_WORDS = 60                                # a classification document = the first 60 words (title + text)
SPLIT_DIR = os.path.join(DATA, 'splits')

SEED = 2026
SEEDS = [17, 29, 43]                          # three seeds for the neural models
VOCAB_SIZE = 32000                            # main tokenizer vocabulary (the same as in Tukeyev and Rysbek)
VOCAB_SWEEP = [8000, 16000, 32000, 48000]
INTRINSIC_TEST_SIZE = 5000                    # sentences of the held-out tokenizer test corpus
SEG_TRAIN_SENTENCES = 50000                   # segmented sentences for the corpus words of FEMSeg-uz

for d in (DATA, CSE_DIR, UD_DIR, CORPUS_DIR, REF_DIR, EXT_DIR, SPLIT_DIR, MODELS, TOK_DIR, SEG_DIR, RESULTS, FIGURES,
          TABLES, PAPER):
    os.makedirs(d, exist_ok=True)
