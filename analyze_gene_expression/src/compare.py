"""Cross-dataset comparison of differential-expression results.

Compares two collapsed-to-gene de_results frames and writes three files
to output_dir: shared_de_genes.csv, log2fc_scatter.png, overlap_summary.txt.
Called by ge.py; not meant to be run directly. Caller is expected to have
already invoked run_diffex(..., collapse_to_gene=True) on each dataset.
"""

from __future__ import annotations
import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)

_DPI = 200
_SAVEFIG_KW = {'dpi': _DPI, 'bbox_inches': 'tight'}
_DEFAULT_ADJ_P_MAX = 0.05
_DEFAULT_ABS_LOG2FC_MIN = 1.0
_ANNOTATE_TOP = 15

_COLOR_BOTH = '#d62728'      # significant in both
_COLOR_ONE = '#ff7f0e'       # significant in one
_COLOR_NEITHER = '#bdbdbd'   # neither

_SHARED_CSV_NAME = 'shared_de_genes.csv'
_SCATTER_PNG_NAME = 'log2fc_scatter.png'
_SUMMARY_TXT_NAME = 'overlap_summary.txt'


"""Compare two collapsed-to-gene de_results frames.

Writes three files to output_dir and returns the joined frame
(inner join on gene_symbol).
"""
def run_compare(
        de_a: pd.DataFrame,
        de_b: pd.DataFrame,
        output_dir: str,
        label_a: str,
        label_b: str,
        *,
        adj_p_max: float = _DEFAULT_ADJ_P_MAX,
        abs_log2fc_min: float = _DEFAULT_ABS_LOG2FC_MIN,
        annotate_top: int=_ANNOTATE_TOP,
) -> pd.DataFrame:
    # validate input cols
    required = {'gene_symbol','log2FC','adj_p_value'}
    for name, df in (('de_a',de_a),('de_b',de_b)):
        missing = required-set(df.columns)

        # check if col is missing
        if missing:
            raise ValueError(
                f"{name} missing columns: {sorted(missing)}"
                f"Expected output of run_diffex(...,collapse_to_gene=True)"
            )

        # check if gene_symbol is na
        if df['gene_symbol'].isna().any():
            raise ValueError(
                f"{name} contains NaN gene_symbol entries"
                f"Passe collapse_to_gene=True to run_diffex"
            )

        # check if gene_symbol is duplicated
        if df['gene_symbol'].duplicated().any():
            dup_n = int(df['gene_symbol'].duplicated().sum())
            raise ValueError(
                f"{name} has {dup_n} duplicate gene_symbol rows"
                f"Pass collapse_to_gene=True to run_diffex"
            )
        
    # inner join on gene_symbol - keep only cols actually used
    cols = ['gene_symbol','log2FC','adj_p_value']
    joined = de_a[cols].merge(
        de_b[cols],
        on = 'gene_symbol',
        how='inner',
        suffixes=(f"_{label_a}",f"_{label_b}"),
    )
    logger.info(
       "joined %d × %d genes -> %d shared on gene_symbol",
       len(de_a), len(de_b), len(joined),
   )
    
    # drop rows with NaN in any of the 4 num cols before corr/sign concordance calcs
    nums = [
        f"log2FC_{label_a}",
        f"log2FC_{label_b}",
        f"adj_p_value_{label_a}",
        f"adj_p_value_{label_b}",
    ]
    joined = joined.dropna(subset=nums).reset_index(drop=True)
    if joined.empty:
        raise ValueError(
            "no genes left after inner join + NaN drop - "
            "check that both inputs are collapsed and overlap on gene_symbol"
        )
    
    # significance & classification setup
    sig_a_full = (de_a['adj_p_value'] < adj_p_max) & (de_a['log2FC'].abs() >= abs_log2fc_min)
    sig_b_full = (de_b['adj_p_value'] < adj_p_max) & (de_b['log2FC'].abs() >= abs_log2fc_min)
    set_a = set(de_a.loc[sig_a_full,'gene_symbol'])
    set_b = set(de_b.loc[sig_b_full,'gene_symbol'])

    joined['sig_a']=(
        (joined[f"adj_p_value_{label_a}"] < adj_p_max) & 
        (joined[f"log2FC_{label_a}"].abs() >= abs_log2fc_min)
    )
    joined['sig_b']=(
        (joined[f"adj_p_value_{label_b}"] < adj_p_max) & 
        (joined[f"log2FC_{label_b}"].abs() >= abs_log2fc_min)
    )

    # write genes significant in both to shared_de_genes.csv
    # #  - sort by combined rank so most concordant is at top
    shared = joined[joined['sig_a'] & joined['sig_b']].copy()
    shared['rank_a'] = shared[f"log2FC_{label_a}"].abs().rank(ascending=False)
    shared['rank_b'] = shared[f"log2FC_{label_b}"].abs().rank(ascending=False)
    shared['combined_rank'] = shared['rank_a'] + shared['rank_b']
    shared = (
        shared.sort_values('combined_rank').drop(columns=['rank_a','rank_b']).reset_index(drop=True)
    )

    # write to file
    os.makedirs(output_dir, exist_ok=True)
    shared.to_csv(
        os.path.join(output_dir, _SHARED_CSV_NAME),
        index=False,
        float_format='%.6g',
    )
    logger.info("wrote %s (%d genes)", _SHARED_CSV_NAME, len(shared))

    # calc pearson corr and spearman corr on joined set
    x = joined[f"log2FC_{label_a}"].to_numpy()
    y = joined[f"log2FC_{label_b}"].to_numpy()
    pearson, _ = pearsonr(x,y)
    spearman, _ = spearmanr(x,y)

    # build 3 z-ordered scatter plot
    fig, ax = plt.subplots(figsize=(7,7))
    neither = ~(joined['sig_a']|joined['sig_b'])
    one = (joined['sig_a'] ^ joined['sig_b'])
    both = joined['sig_a'] & joined['sig_b']

    ax.scatter(x[neither], y[neither], s=6, alpha=0.5, 
               c=_COLOR_NEITHER, label=f"neither (n={int(neither.sum())})")
    ax.scatter(x[one], y[one], s=8, alpha=0.7,
               c=_COLOR_ONE, label=f"one only (n={int(one.sum())})")
    ax.scatter(x[both],y[both],s=10,alpha=0.85,
               c=_COLOR_BOTH, label=f"both (n={int(both.sum())})")
    # add reference lines & 1:1 diagonal for dir and mag visuals
    lim = float(max(np.abs(x).max(),np.abs(y).max()))*1.05
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.axhline(0,color='grey',linewidth=0.6,linestyle='--')
    ax.axvline(0,color='grey',linewidth=0.6,linestyle='--')
    ax.plot([-lim,lim],[-lim,lim],color='grey',linewidth=0.6,linestyle=':')

    # annotate top concordant genes w smallest combined_rank among both-sig set
    if annotate_top > 0 and not shared.empty:
        top = shared.head(annotate_top)
        for _,row in top.iterrows():
            ax.annotate(
                str(row['gene_symbol']),
                xy=(row[f"log2FC_{label_a}"],row[f"log2FC_{label_b}"]),
                xytext=(4,4), textcoords='offset points',
                fontsize=7,
            )
    
    # fig labeling
    ax.set_xlabel(f"log2FC ({label_a})")
    ax.set_ylabel(f"log2FC ({label_b})")
    ax.set_title(
        f'log2FC concordance: Pearson r={pearson:.3f}, '
        f"Spearman ρ={spearman:.3f} "
        f"(n={len(joined)})"
    )
    ax.legend(loc='upper left',fontsize=8)

    # save fig
    fig.savefig(os.path.join(output_dir, _SCATTER_PNG_NAME),**_SAVEFIG_KW)
    plt.close(fig)
    logger.info("wrote %s",_SCATTER_PNG_NAME)

    # build & write summary text
    a = set_a - set_b
    b = set_b - set_a
    both = set_a & set_b
    union = set_a | set_b
    jaccard = len(both) / len(union) if union else 0.0

    if both:
        signs_a = np.sign(de_a.set_index('gene_symbol').loc[list(both),'log2FC'])
        signs_b = np.sign(de_b.set_index('gene_symbol').loc[list(both),'log2FC'])
        same_sign = int((signs_a == signs_b).sum())
        concordance = same_sign / len(both)
    else:
        same_sign = 0
        concordance = float('nan')
    lines = [
        f"comparison: {label_a} vs {label_b}",
        f"thresholds: adj_p < {adj_p_max}, |log2FC| >= {abs_log2fc_min}",
        f"genes joined on gene_symbol (inner): {len(joined)}",
        "",
        f"significant in {label_a}: {len(set_a)}",
        f"significant in {label_b}: {len(set_b)}",
        f"significant in both:    {len(both)}",
        f"only in {label_a}:       {len(a)}",
        f"only in {label_b}:       {len(b)}",
        "",
        f"jaccard:               {jaccard:.4f}",
        f"sign concordance:      {concordance:.4f} ({same_sign}/{len(both)})",
        f"pearson r (joined):    {pearson:.4f}",
        f"spearman rho (joined): {spearman:.4f}",
    ]

    summary_path = os.path.join(output_dir, _SUMMARY_TXT_NAME)
    with open(summary_path, 'w', encoding='utf8') as f:
        f.write('\n'.join(lines)+'\n')
    logger.info("wrote %s", _SUMMARY_TXT_NAME)

    return joined
        

    