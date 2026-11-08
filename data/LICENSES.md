# Licenses of the data files

| Path | Source | License |
|---|---|---|
| `data/cse/uzmorphanalyser_0.0.22/` | UzMorphAnalyser, Python package version 0.0.22 (Salaev, 2024), files `affixes.csv`, `non_affixed_stems.csv`, `number_stems.csv`, `exception_stems.csv` | MIT |
| `data/cse/cse_morphs.tsv` | division of `affixes.csv` into morphemes (this work) | MIT |
| `data/reference/` | derived from UD_Uzbek-UzUDT, Universal Dependencies release 2.18 (Matlatipov and Aripov, LREC 2026) | CC BY-SA 4.0 |
| `data/splits/` | document ids of the Uzbek News Dataset v0.0.1 (Kuriyozov et al., 2023), doi:10.5281/zenodo.7677431 | CC BY 4.0 |
| `data/cse/stems.csv` (built in step 1, not included) | one-word entries of apertium-uzb, commit 1bdee4a | GPL-3.0 |
| `data/corpus/`, `data/extrinsic/` (built in steps 1-2, not included) | texts of the Uzbek News Dataset | CC BY 4.0 |

The external files are downloaded by `scripts/s00_download_data.py` from their original sources and checked against
fixed checksums.
