"""Clustered heatmap of top differentially expressed genes.

Renders a sample × gene heatmap with hierarchical clustering on both axes
and a per-sample group color bar. Called by ge.py; not meant to be run
directly. Caller is expected to have already invoked load_geo_dataset,
assign_groups, and run_diffex.
"""

from __future__ import annotations
import logging
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from shared_utils import top_de_genes, normalize

logger = logging.getLogger(__name__)

_DPI = 200
_SAVEFIG_KW = {'dpi': _DPI, 'bbox_inches': 'tight'}
_GROUP_PALETTE = 'Set2'
_CMAP = 'RdBu_r'

"""Render clustered heatmap of top DE genes to output_path.
    Defaults: top 50 genes meeting p<0.05 and |log2FC|>=1."""
def plot_heatmap(
        expression: pd.DataFrame,
        samples: pd.DataFrame,
        de_results: pd.DataFrame,
        output_path: str,
        top: int = 50,
        adj_p_max: float = 0.05,
        abs_log2fc_min: float = 1.0,
) -> None:
    
    # validate input cols
    required_de = {'log2FC','adj_p_value','probe_id'}
    missing = required_de - set(de_results.columns)
    if missing:
        raise ValueError(
            f"de_results missing columns: {sorted(missing)}"
            f"Expected output of shared_utils.differential_expression"
        )
    if 'group' not in samples.columns:
        raise ValueError(
            "samples missing 'group' column"
            "call shared_utils.assign_groups before plot_heatmap"
        )
    
    # pick gene set - default top 50
    top_genes = top_de_genes(
        de_results,
        n=top,
        adj_p_max=adj_p_max,
        abs_log2fc_min=abs_log2fc_min,
    )
    if top_genes.empty:
        logger.warning(
            "no genes past thresholds (p<%s, |log2FC|>=%s); skipping heatmap",
            adj_p_max,
            abs_log2fc_min,
        )
        return
    # drop unassigned samples & realign cols
    labeled = samples.dropna(subset=['group'])
    if labeled.empty:
        raise ValueError("no samples have an assigned group")
    probe_ids = top_genes['probe_id'].tolist()
    mat = expression.loc[probe_ids,labeled.index]

    # drop 0-variance rows before z-scoring so clustermap doesnt crash during linkage
    row_std = mat.std(axis=1)
    keep = row_std > 0
    if not keep.all():
        dropped = (~keep).sum()
        logger.warning("dropping %d 0-variance rows before z-score",dropped)
        mat = mat.loc[keep]

    # z-score per row
    z = normalize(mat,'zscore')

    # build col color bar
    groups = labeled.loc[z.columns,'group']
    group_levels = sorted(groups.unique())
    palette = sns.color_palette(_GROUP_PALETTE, len(group_levels))
    group_to_color = dict(zip(group_levels, palette))
    col_colors = groups.map(group_to_color)
    col_colors.name='group'

    # relable probe_ids row to gene symbol where available
    if 'gene_symbol' in top_genes.columns:
        label_map = top_genes.set_index('probe_id')['gene_symbol']
        new_index=[
            label_map.get(p) if isinstance(label_map.get(p),str) and 
                label_map.get(p) != ''
            else p
            for p in z.index
        ]
        z = z.copy()
        z.index = new_index

    # build clustermap
    height = max(6.0, min(20.0, top * 0.18))
    g = sns.clustermap(
        z,
        cmap = _CMAP,
        center = 0,
        col_colors=col_colors,
        figsize=(10,height),
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=(0.12,0.18),
        cbar_pos=(0.02,0.85,0.03,0.1),
        linewidths=0,
    )
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6,
        )
    g.ax_heatmap.set_yticklabels(
            g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=7,
        )
    g.ax_heatmap.set_xlabel('sample')
    g.ax_heatmap.set_ylabel('gene')
    g.fig.suptitle(
        f'Top {len(z)} DE genes (z-scored rows) — '
        f'{len(group_levels)} groups',
        y=1.02, fontsize=10,
        )
    
    # save fig
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    g.savefig(output_path,**_SAVEFIG_KW)
    plt.close(g.figure)
    logger.info(
        "wrote %s (genes=%d, samples=%d, groups=%d)",
        output_path, z.shape[0], z.shape[1], len(group_levels),
    )
