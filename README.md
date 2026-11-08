# UzMorphSegTok

Code, data splits, trained models and result files of the paper

> Ulugbek Salaev. **CSE-Constrained Neural Morphological Segmentation and Morphology-Aware Tokenization for Uzbek.**
> Manuscript submitted to *Information* (MDPI), 2026.

The repository contains everything needed to repeat the experiments of the paper on a CPU: the scripts download the
two data sources and one knowledge resource at fixed versions, check their checksums, rebuild all derived data
deterministically, train the models and recompute every table and figure.

## What is in the repository

| Folder | Content |
|---|---|
| `uzseg/` | the library: text normalization, the CSE table and its division into morphemes, the sets of allowed segmentations Ω(w), the rule-based segmenters CSE-Seg and CSE-Seg+P, the neural segmenter FEMSeg-uz, tokenizers, evaluation measures |
| `scripts/` | the experiment steps (`s00` ... `s10`), described below |
| `data/cse/uzmorphanalyser_0.0.22/` | the CSE table (`affixes.csv`) and three stem lists of the UzMorphAnalyser package, version 0.0.22 (MIT license) |
| `data/splits/` | the document ids of every part of the news dataset (tokenizer training and test corpus, classification training, development and test data) |
| `data/reference/` | the reference segmentation derived from UD_Uzbek-UzUDT r2.18 (CC BY-SA 4.0, like the treebank) |
| `data/checksums.json` | SHA-256 checksums of all prepared data files |
| `models/femseg/` | 24 FEMSeg-uz models: 8 configurations x 3 seeds (17, 29, 43) |
| `models/tokenizers/` | 18 BPE and Unigram tokenizers (HuggingFace Tokenizers JSON files) |
| `models/*.bin` | Morfessor models |
| `results/` | all result files used in the paper (JSON) |
| `paper/figures.py` | draws the data figures of the paper from `results/` |

## External data (downloaded by step 0)

| Resource | Version | License | Use in the paper |
|---|---|---|---|
| UD_Uzbek-UzUDT (Matlatipov and Aripov, LREC 2026) | UD release 2.18, commit `170a1b3` | CC BY-SA 4.0 | reference set: official train part (483 sentences) for gold training words and development, official test part (201 sentences) for testing |
| Uzbek News Dataset (Kuriyozov et al., 2023) | v0.0.1, [doi:10.5281/zenodo.7677431](https://doi.org/10.5281/zenodo.7677431) | CC BY 4.0 | tokenizer corpus, paradigm statistics, corpus words of FEMSeg-uz, classification data |
| apertium-uzb stem lexicon (Apertium project) | commit `1bdee4a` | GPL-3.0 | lexical support of candidate segmentations |

The UzMorphAnalyser files are part of this repository. The news dataset is not redistributed: step 1 rebuilds every
text file from the downloaded archive, and `scripts/verify_release.py` checks that the result is byte-identical to
the data used in the paper.

## Installation

Python 3.12 and the packages in `requirements.txt` (CPU versions are enough):

```
pip install -r requirements.txt
```

## Checking the released models (about 40 minutes)

```
python run_all.py 0 2                 # download, prepare and segment the data
python scripts/verify_release.py      # compare checksums and recompute the scores of the released models
```

The second command prints `same` or `DIFFERENT` for every prepared data file and for the scores of CSE-Seg,
CSE-Seg+P, the main tokenizers and all 24 FEMSeg-uz models.

## Repeating all experiments

```
python run_all.py                     # steps 0-10
```

| Step | Script | What it does |
|---|---|---|
| 0 | `s00_download_data.py` | downloads UzUDT r2.18, the news dataset and the apertium-uzb lexicon, checks the checksums |
| 1 | `s01_prepare_data.py` | stem lexicon, morpheme division of the CSE table, reference set, filtered news documents, fixed splits, tokenizer corpus, classification data |
| 2 | `s02_segment_corpus.py` | paradigm statistics, selection of beta and gamma on the UzUDT train part, segmentation of the corpus by CSE-Seg+P |
| 3 | `s03_train_tokenizers.py` | BPE and Unigram tokenizers on the raw and the segmented corpus, V = 8K, 16K, 32K, 48K |
| 4 | `run_femseg_all.py` | FEMSeg-uz: 8 configurations x 3 seeds, and five-fold cross-validation of the main model |
| 5 | `s06_eval_segmenters.py` | Morfessor, Morfessor+UD, UzMorphAnalyser, CSE-Seg, CSE-Seg+P, CSE-Seg+P+UD |
| 6 | `s08_femseg_tokenizers.py` | tokenizers on the corpus segmented by FEMSeg-uz |
| 7 | `s05_eval_tokenizers.py`, `s09_examples.py` | tokenizer statistics on held-out text, agreement with morpheme boundaries, examples |
| 8 | `s07_extrinsic.py`, `check_models.py` | news classification (11 tokenizers x 3 seeds; BPE-raw and BPE-CSE+seg also with 5,000 training documents per category), check of the saved models |
| 9 | `s10_analysis.py` | bootstrap confidence intervals, McNemar tests, stem check, seen and unseen words, cross-validation |
| 10 | `paper/figures.py` | figures |

All random choices use fixed seeds (2026 for data splits, 17, 29 and 43 for the neural models). The full run takes
about 35 hours on a laptop with a four-core Intel Core i5-1135G7 processor and 8 GB of memory, without a GPU.

## Using the segmenters

The segmenters need the stem lexicon and the paradigm statistics, which steps 0-2 create (`python run_all.py 0 2`).

```python
import sys; sys.path.insert(0, '.')
from uzseg.segmenters import CSESegmenter, FEMSegmenter

fem = FEMSegmenter('full_s17')          # FEMSeg-uz, CSE-constrained decoding
rule = CSESegmenter()                   # CSE-Seg+P
for w in ['kitoblarimizda', "o'qituvchilarimizning"]:
    print(w, sorted(fem(w)), sorted(rule(w)))   # boundary positions inside the word
# kitoblarimizda [5, 8, 12] [5, 8, 12]                  kitob+lar+imiz+da
# o'qituvchilarimizning [10, 13, 17] [10, 13, 17]       o'qituvchi+lar+imiz+ning
```

Words are given in lower case with the plain apostrophe (`o'`, `g'`).

## Main results (UzUDT test part)

| System | Boundary F1 (%) | 95% CI | Exact match (%) |
|---|---|---|---|
| Morfessor | 65.03 | 62.56-67.52 | 59.60 |
| Morfessor+UD | 80.12 | 77.75-82.24 | 77.90 |
| UzMorphAnalyser 0.0.22 | 89.58 | 87.98-91.09 | 85.29 |
| CSE-Seg | 89.33 | 87.75-90.89 | 85.73 |
| CSE-Seg+P | 92.97 | 91.77-94.12 | 90.72 |
| CSE-Seg+P+UD | 93.85 | 92.64-94.99 | 91.92 |
| FEMSeg-uz (mean of 3 seeds) | 94.67 | 93.59-95.69 | 92.91 |

Five-fold cross-validation over all 684 sentences: FEMSeg-uz 95.53, CSE-Seg+P+UD 94.87, CSE-Seg+P 92.97 (boundary F1, %).

## Licenses

The code is released under the MIT license (`LICENSE`). The data files keep the licenses of their sources:
`data/reference/` CC BY-SA 4.0 (derived from UD_Uzbek-UzUDT), `data/splits/` CC BY 4.0 (ids of the Uzbek News
Dataset), `data/cse/uzmorphanalyser_0.0.22/` MIT (UzMorphAnalyser). The stem lexicon `data/cse/stems.csv`, which is
built in step 1 from apertium-uzb, is GPL-3.0 and is therefore not included; step 0 downloads its source file.

## Citation

```
@article{salaev2026uzmorphsegtok,
  author  = {Salaev, Ulugbek},
  title   = {CSE-Constrained Neural Morphological Segmentation and Morphology-Aware Tokenization for Uzbek},
  journal = {Information},
  year    = {2026},
  note    = {Submitted}
}
```
