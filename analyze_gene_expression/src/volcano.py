"""Volcano plot for differential-expression results.

Renders log2FC (x) vs -log10(adj_p_value) (y) as a static PNG, classifying
points as up-regulated, down-regulated, or non-significant against caller-
supplied thresholds. Called by ge.py; not meant to be run directly.
"""

from __future__ import annotations
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

_DPI = 200
_SAVEFIG_KW = {'dpi': _DPI, 'bbox_inches': 'tight'}
_P_FLOOR = 1e-300       # clip for adj_p_value before -log10
_POINT_SIZE = 8
_POINT_ALPHA = 0.6
_COLOR_NONSIG = '#bdbdbd'
_COLOR_UP = '#d62728'
_COLOR_DOWN = '#1f77b4'

"""Render volcano plot to output_path. Defaults: p<0.05, |log2FC|>=1, top-10 labels."""
def plot_volcano(
        de_results: pd.DataFrame,
        output_path: str,
        adj_p_max: float = 0.05,
        abs_log2fc_min: float = 1.0,
        annotate_top: int = 10,
) -> None:
    # validate input cols
    required = {'log2FC', 'adj_p_value'}
    missing = required-set(de_results.columns)
    if missing:
        raise ValueError(
            f'de_results missing cols: {sorted(missing)}'
            f'Expected output of shared_utils.differential_expression'
        )
    # compute axis - drop rows with NaN p-values 
    df = de_results.dropna(subset=['adj_p_value','log2FC']).copy()
    x = df['log2FC'].to_numpy()
    p_clipped = df['adj_p_value'].clip(lower=_P_FLOOR)
    y = -np.log10(p_clipped.to_numpy())

    # classify rows into [up, down, non-significant]
    sig = df['adj_p_value'] < adj_p_max
    up = sig & (df['log2FC']>=abs_log2fc_min)
    down = sig & (df['log2FC'] <= -abs_log2fc_min)
    non_sig = ~(up | down)

    # build volcano fig
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(x[non_sig],y[non_sig],s=_POINT_SIZE, alpha=_POINT_ALPHA,
               c=_COLOR_NONSIG, label='non-significant')
    ax.scatter(x[up],y[up],s=_POINT_SIZE,alpha=_POINT_ALPHA,
               c=_COLOR_UP,label=f'up (n={int(up.sum())})')
    ax.scatter(x[down],y[down],s=_POINT_SIZE, alpha=_POINT_ALPHA,
               c=_COLOR_DOWN, label=f'down (n={int(down.sum())})')
    # create threshold reference lines
    ax.axhline(-np.log10(adj_p_max), color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(abs_log2fc_min, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(-abs_log2fc_min, color='grey', linestyle='--', linewidth=0.8)

    # top N annotation: pick annotate_top rows with smallest adj_p_value in significant genes
    if annotate_top > 0:
        label_col = 'gene_symbol' if 'gene_symbol' in df.columns else 'probe_id'
        sig_dif = df[up | down].copy()
        sig_dif['_y'] = -np.log10(sig_dif['adj_p_value'].clip(lower=_P_FLOOR))
        top = sig_dif.nsmallest(annotate_top, 'adj_p_value')

        for _, row in top.iterrows():
            label = row.get(label_col)
            if pd.isna(label) or label=='':
                label = row.get('probe_id','')
            ax.annotate(
                str(label),
                xy=(row['log2FC'],row['_y']),
                xytest=(4,4),
                textcoords='offset points',
                fontsize=7,
            )

    ax.set_xlabel('log2 fold change (group_a - group_b)')
    ax.set_ylabel('-log10(adj_p_value)')
    ax.set_title(
        f'Volcano: {int(up.sum())} up, {int(down.sum())} down '
        f'(p<{adj_p_max}), |log2FC|>={abs_log2fc_min}'
    )
    ax.legend(loc='upper right', fontsize=8)

    # output fig
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, **_SAVEFIG_KW)
    plt.close(fig)
    logger.info("wrote %s (up=%d, down=%d, n=%d)",output_path, int(up.sum()),int(down.sum()),len(df))