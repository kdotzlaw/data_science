"""
EDA for a single GEO dataset

Writes 1 text summary & 4 PNG plots to results/<accession>/
Called by ge.py -- caller expected to have already invoked shared_utils.load_geo_dataset()
 and assign_groups before passing expression and samples in

"""
from __future__ import annotations
import logging
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from shared_utils import detect_log_scale

logger = logging.getLogger(__name__)

_DPI = 200
_SAVEFIG_KW = {'dpi': _DPI, 'bbox_inches': 'tight'}
_MAX_BOXPLOT_SAMPLES = 40


# write summary to txt file --> output to /results
def _write_summary(expression, samples, output_dir):
    # compute scalars
    n_genes, n_samples = expression.shape
    values = expression.values
    value_min = float(np.nanmin(values))
    value_max = float(np.nanmax(values))
    value_median = float(np.nanmedian(values))
    log_scale = detect_log_scale(expression)

    # per-sample NaN fraction
    nan_fraction = expression.isna().mean(axis=0) # series, 1 entry per sample
    nan_min = float(nan_fraction.min())
    nan_median = float(nan_fraction.median())
    nan_max = float(nan_fraction.max())

    # group counts -- if caller assigned groups before calling run_eda
    if 'group' in samples.columns:
        group_counts = samples['group'].value_counts(dropna=False).to_dict()
    else:
        group_counts=None

    # build output list
    lines = [
        f"Genes (rows): {n_genes}",
        f"Samples (cols): {n_samples}",
        f"Value range: [{value_min:.3f},{value_max:.3f}]",
        f"Global median: {value_median:.3f}",
        f"Detected log-scale: {log_scale}",
        f"Per-sample NaN fraction: min={nan_min:.4f}, median={nan_median:.4f}, max={nan_max:.4f}",
    ]
    # add group count
    if group_counts is not None:
        lines.append(f"Group counts: {group_counts}")
    else:
        lines.append("Group counts: <not assigned>")

    # write to given dir
    path = os.path.join(output_dir,'summary.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')
    logger.info("wrote %s (%d genes x %d samples)", path, n_genes, n_samples)

# create box plots --> output to /results
def _plot_sample_boxplots(expression, output_dir):
    # get shape
    n = expression.shape[1]

    # get boxplot samples
    sub = expression.iloc[:,:_MAX_BOXPLOT_SAMPLES if n>_MAX_BOXPLOT_SAMPLES else expression]

    # plot
    fig, ax = plt.subplots(figsize=(max(6,sub.shape[1]*0.25),5))
    sns.boxplot(data=sub,ax=ax,fliersize=1,linewidth=0.5)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90,fontsize=7)
    ax.set_ylabel('Expression')
    ax.set_title(f'Per-sample expression distribution (first {sub.shape[1]} of {n})')

    # save to /result
    fig.savefig(os.path.join(output_dir,'sample_boxplots.png'), **_SAVEFIG_KW)
    plt.close(fig)

# create correlation heatmap --> output to /results
def _plot_correlation_heatmap(expression, samples, output_dir):
    # sample-sample correlation
    corr = expression.corr()

    # check for 'group' in samples for ordered corr
    if 'group' in samples.columns:
        order = samples.dropna(subset=['group']).sort_values('group').index
        order = [s for s in order if s in corr.index]
        corr = corr.loc[order,order]
    # plot heatmat
    fig, ax = plt.subplots(figsize=(max(6,corr.shape[0]*0.15),max(5,corr.shape[0]*0.15)))
    sns.heatmap(corr,cmap='viridis',square=True,ax=ax,
                 xticklabels=False, yticklabels=False,
                 cbar_kws={'label':'Pearson r'})
    ax.set_title('Sample-Sample Correlation')
    fig.savefig(os.path.join(output_dir,'sample_correlation_heatmap.png'), **_SAVEFIG_KW)
    plt.close(fig)

# create  pca plot --> output to /results
def _plot_pca(expression, samples, output_dir):
    # drop na
    mat = expression.dropna(axis=0,how='any')
    if mat.empty:
        logger.warning("PCA skipped: no genes without NaN")
        return

    # transpose expression to get sample pca
    X = StandardScaler().fit_transform(mat.T.values) # samples as rows
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    # get variance
    var = pca.explained_variance_ratio_*100

    # plot pca
    fig, ax = plt.subplots(figsize=(7,6))

    # if there are groups, plot like this
    if 'group' in samples.columns:
        for grp, sub in samples.groupby('group'):
            index = [samples.index.get_loc(s) for s in sub.index if s in samples.index]
            ax.scatter(coords[index,0],coords[index,1],label=str(grp),s=30,alpha=0.8)
        ax.legend(title='group')
    else:
        ax.scatter(coords[:,0],coords[:,1],s=30,alpha=0.8)
    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
    ax.set_title("PCA of samples")

    fig.savefig(os.path.join(output_dir,'pca_scatter.png'),**_SAVEFIG_KW)
    plt.close(fig)

# create gene variance plot --> output to /results
def _plot_gene_variance(expression, output_dir):
    # get variance
    var = expression.var(axis=1).dropna()

    # plot
    fig,ax = plt.subplots(figsize=(7,4))
    ax.hist(var,bins=80)
    ax.set_xlabel('Per-gene variance')
    ax.set_ylabel('Gene count')
    # use log scale to help hist be less right skewed
    ax.set_yscale('log')
    ax.set_title('Per-gene Variance Distribution')

    fig.savefig(os.path.join(output_dir,'gene_variance_hist.png'),**_SAVEFIG_KW)
    plt.close(fig)

# Public entry function
def run_eda(expression: pd.DataFrame, samples:pd.DataFrame, output_dir: str)->None:
    # put EDA output into output_dir -- create if needed
    os.makedirs(output_dir,exist_ok=True)
    logger.info("Eda on %d genes x %d samples -> %s",
                expression.shape[0], expression.shape[1],output_dir)
    # summary
    _write_summary(expression, samples, output_dir)

    # boxplots
    _plot_sample_boxplots(expression, output_dir)

    # correlation
    _plot_correlation_heatmap(expression,samples,output_dir)

    # pca
    _plot_pca(expression,samples,output_dir)

    # gene variance
    _plot_gene_variance(expression,output_dir)