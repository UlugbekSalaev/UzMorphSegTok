"""Data figures of the paper, drawn with Matplotlib from the result files.

  fig3_segmenters.pdf  boundary F1 and exact match of the segmenters with 95% bootstrap intervals;
  fig4_vocab.pdf       whole-text boundary F1 and tokens per word as functions of the vocabulary size;
  fig5_extrinsic.pdf   macro-F1 of the news classifier and the number of tokens per training document.
Figures 1 and 2 are diagrams drawn in Visio (paper/figures_visio.py).
Usage: python paper/figures.py
"""
import json
import os
import statistics as st
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as C  # noqa: E402

OUT = os.path.join(HERE, 'MDPI_2026_UzMorphSegTok', 'figures')    # folder of the LaTeX sources of the paper
if not os.path.isdir(os.path.dirname(OUT)):
    OUT = os.path.join(HERE, 'figures')
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
TEST = C.TEST_SETS[0]

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.6,
    'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'xtick.major.size': 2.5,
    'ytick.major.size': 2.5, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.dpi': 300,
})


def load(*path):
    return json.load(open(os.path.join(C.RESULTS, *path), encoding='utf-8'))


def grid(ax, axis='both'):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


SEG_ORDER = [('Morfessor', 'Morfessor'), ('Morfessor+UD', 'Morfessor+UD'), ('UzMorphAnalyser', 'UzMorphAnalyser'),
             ('CSE-Seg', 'CSE-Seg'), ('CSE-Seg+P', 'CSE-Seg+P'), ('CSE-Seg+P+UD', 'CSE-Seg+P+UD'),
             ('FEMSeg-uz full free', 'FEMSeg-uz, free'), ('FEMSeg-uz full hard', 'FEMSeg-uz, hard')]


def fig_segmenters():
    ci = load('analysis.json')['segmenters_ci']
    rows = [(label, ci[key]) for key, label in SEG_ORDER if key in ci]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.35), sharey=True)
    for ax, metric, title in ((axes[0], 'f1', 'Boundary F1 (%)'), (axes[1], 'em', 'Exact match (%)')):
        for i, (label, e) in enumerate(rows):
            main = label == 'FEMSeg-uz, hard'
            color = BLUE if main else MUTED
            lo, hi = e[metric + '_ci']
            ax.plot([lo, hi], [i, i], color=color, linewidth=1.2, solid_capstyle='butt')
            ax.plot(e[metric], i, 'o', color=color, markersize=5, markeredgecolor='white', markeredgewidth=0.8)
        ax.set_xlabel(title)
        grid(ax, 'x')
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[0] for r in rows])
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    fig.tight_layout(w_pad=1.5)
    save(fig, 'fig3_segmenters')


SWEEP = [('bpe_raw_{}', 'BPE-raw', BLUE, 'o', True), ('uni_raw_{}', 'Uni-raw', BLUE, 's', False),
         ('bpe_cse_{}', 'BPE-CSE', ORANGE, 'o', True), ('uni_cse_{}', 'Uni-CSE', ORANGE, 's', False),
         ('bpe_cse_{}+seg', 'BPE-CSE+seg', AQUA, 'o', True), ('uni_cse_{}+seg', 'Uni-CSE+seg', AQUA, 's', False)]


def fig_vocab():
    tk = load('tokenizers_intrinsic.json')
    sizes = C.VOCAB_SWEEP
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.5))
    for fmt, label, color, marker, filled in SWEEP:
        names = [fmt.format(v) for v in sizes]
        if not all(n in tk for n in names):
            continue
        f1 = [100 * tk[n]['boundary'][TEST]['text_f1'] for n in names]
        tpw = [tk[n]['intrinsic']['tokens_per_word'] for n in names]
        style = dict(color=color, marker=marker, markersize=4.5, linewidth=1.3,
                     markerfacecolor=color if filled else 'white', markeredgecolor=color, markeredgewidth=1.0,
                     linestyle='-' if filled else (0, (4, 2)))
        axes[0].plot(sizes, f1, label=label, **style)
        axes[1].plot(sizes, tpw, label=label, **style)
    for ax, ylabel in ((axes[0], 'Whole-text boundary F1 (%)'), (axes[1], 'Tokens per word')):
        ax.set_xscale('log')
        ax.set_xticks(sizes)
        ax.set_xticklabels([f'{v // 1000}K' for v in sizes])
        ax.minorticks_off()
        ax.set_xlabel('Vocabulary size')
        ax.set_ylabel(ylabel)
        grid(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02),
               handlelength=2.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=2.0)
    save(fig, 'fig4_vocab')


# groups of Table 11 with the colours of Figure 4; grey: multilingual tokenizers
EXT_ROWS = [('Trained on raw text', BLUE, [('bpe_raw_32000', 'BPE-raw'), ('uni_raw_32000', 'Uni-raw')]),
            ('CSE-Seg+P output, raw mode', ORANGE, [('bpe_cse_32000', 'BPE-CSE'), ('uni_cse_32000', 'Uni-CSE')]),
            ('Segmented mode', AQUA, [('bpe_cse_32000+seg', 'BPE-CSE+seg'), ('uni_cse_32000+seg', 'Uni-CSE+seg'),
                                      ('bpe_fem_32000+seg', 'BPE-FEM+seg'), ('uni_fem_32000+seg', 'Uni-FEM+seg')]),
            ('Multilingual', MUTED, [('mT5', 'mT5'), ('XLM-R', 'XLM-R'), ('mBERT', 'mBERT')])]


def fig_extrinsic():
    """Macro-F1 with 95% intervals (left) and tokens per training document (right), one row per tokenizer."""
    ci = load('analysis.json')['classification_ci']
    runs = {}
    for f in os.listdir(os.path.join(C.RESULTS, 'extrinsic')):
        if f.endswith('.json'):
            r = load('extrinsic', f)
            runs.setdefault(r['tokenizer'], []).append(r)
    rows = []                                     # (label, colour, tokenizer); a group title has no tokenizer
    for title, color, items in EXT_ROWS:
        rows.append((title, None, None))
        rows += [(label, color, name) for name, label in items if name in runs and name in ci]
    ys = list(range(len(rows) - 1, -1, -1))
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0), sharey=True, gridspec_kw={'width_ratios': [1.3, 1]})
    for y, (label, color, name) in zip(ys, rows):
        if name is None:
            continue
        f1 = st.mean(100 * r['test_macro_f1'] for r in runs[name])
        lo, hi = ci[name]['macro_f1_ci']
        dot = dict(color=color, markersize=5, markeredgecolor='white', markeredgewidth=0.8)
        axes[0].plot([lo, hi], [y, y], color=color, linewidth=1.2, solid_capstyle='butt')
        axes[0].plot(f1, y, 'o', **dot)
        axes[1].plot(runs[name][0]['mean_tokens'], y, 'o', **dot)
    for ax, xlabel in ((axes[0], 'Test macro-F1 (%)'), (axes[1], 'Tokens per training document')):
        ax.set_xlabel(xlabel)
        grid(ax, 'x')
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([label for label, _, _ in rows])
    for tick, (_, _, name) in zip(axes[0].get_yticklabels(), rows):
        if name is None:
            tick.set_fontstyle('italic')
            tick.set_color(MUTED)
    axes[0].set_ylim(-0.6, len(rows) - 0.4)
    fig.tight_layout(w_pad=1.5)
    save(fig, 'fig5_extrinsic')


def main():
    fig_segmenters()
    fig_vocab()
    fig_extrinsic()
    print('saved:', OUT)


if __name__ == '__main__':
    main()
