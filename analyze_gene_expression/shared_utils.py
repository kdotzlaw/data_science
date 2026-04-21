"""Shared helpers for GEO gene-expression analysis.

Kept deliberately narrow: loading, group assignment, normalization, diff-
expression, top-gene selection, plus a couple of small utilities
(`detect_log_scale`, `subset_by_group`) reused across `src/` modules.

All functions are pure - no file I/O except `load_geo_dataset` (which caches
to disk) and no mutation of inputs (all transforms return new frames).
"""

from __future__ import annotations
import logging
import os
from typing import Callable

import numpy as np
import pandas as pd
import GEOparse
from scipy import stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

_PREFERRED_SAMPLE_COLS = ['title', 'source_name_ch1', 'characteristics_ch1', 'platform_id']
_ANNOTATION_RENAME = {
    'Gene Symbol': 'gene_symbol',
    'Gene Title': 'gene_title',
    'ENTREZ_GENE_ID': 'entrez_id',
}


'''
---LOAD_GEO_DATASET---
INPUT: accession (str), cache_dir (str)
PROCESS: download/parse GEO series
OUTPUT: (expression, samples, annotation) tuple of dataframes
'''
def load_geo_dataset(
    accession: str,
    cache_dir: str = 'data',
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    os.makedirs(cache_dir, exist_ok=True)

    try:
        gse = GEOparse.get_GEO(geo=accession, destdir=cache_dir, silent=True)
    except Exception as exc:
        cache_file = os.path.join(cache_dir, f'{accession}_family.soft.gz')
        if os.path.exists(cache_file):
            logger.warning(f'{accession}: cache appears corrupted, retrying download')
            os.remove(cache_file)
            try:
                gse = GEOparse.get_GEO(geo=accession, destdir=cache_dir, silent=True)
            except Exception as retry_exc:
                raise RuntimeError(f'Failed to load {accession}: {retry_exc}') from retry_exc
        else:
            raise RuntimeError(f'Failed to load {accession}: {exc}') from exc

    expression = gse.pivot_samples('VALUE')
    expression = expression.apply(pd.to_numeric, errors='coerce')
    expression = expression.dropna(how='all')
    if expression.empty:
        raise ValueError(f'{accession}: expression matrix is empty after NaN filtering')

    sample_records = {}
    for gsm_id, gsm in gse.gsms.items():
        flat = {k: ' | '.join(str(x) for x in v) for k, v in gsm.metadata.items()}
        sample_records[gsm_id] = flat
    samples = pd.DataFrame.from_dict(sample_records, orient='index')
    samples.index.name = 'gsm_id'

    leading = [c for c in _PREFERRED_SAMPLE_COLS if c in samples.columns]
    remaining = sorted(c for c in samples.columns if c not in leading)
    samples = samples[leading + remaining]

    platforms = list(gse.gpls.values())
    if len(platforms) > 1:
        logger.warning(
            f'{accession} has {len(platforms)} platforms; using {platforms[0].name}'
        )
    annotation = platforms[0].table.copy()
    if 'ID' in annotation.columns:
        annotation = annotation.set_index('ID')
    else:
        first_col = annotation.columns[0]
        logger.warning(f'{accession}: no "ID" column in annotation, using {first_col!r}')
        annotation = annotation.set_index(first_col)
    annotation = annotation.rename(columns=_ANNOTATION_RENAME)
    if 'gene_symbol' not in annotation.columns:
        logger.warning(
            f'{accession}: annotation has no gene_symbol column; '
            'downstream plots will fall back to probe IDs'
        )

    return expression, samples, annotation


'''
---DETECT_LOG_SCALE---
INPUT: expression (DataFrame)
PROCESS: median/max heuristic (log-scaled data has median<30 and max<100)
OUTPUT: bool
'''
def detect_log_scale(expression: pd.DataFrame) -> bool:
    values = expression.values
    median = np.nanmedian(values)
    maximum = np.nanmax(values)
    return bool(median < 30 and maximum < 100)


'''
---ASSIGN_GROUPS---
INPUT: samples (DataFrame), group_map (dict | callable), source_col (str), substrings (dict)
PROCESS: add 'group' column via dict lookup, callable, or substring match
OUTPUT: new samples DataFrame with 'group' column
'''
def assign_groups(
    samples: pd.DataFrame,
    group_map: dict[str, str] | Callable[[pd.Series], str | None] | None = None,
    source_col: str | None = None,
    substrings: dict[str, str] | None = None,
) -> pd.DataFrame:
    substring_mode = source_col is not None and substrings is not None
    if sum([isinstance(group_map, dict), callable(group_map), substring_mode]) != 1:
        raise ValueError(
            'assign_groups: supply exactly one of (group_map dict), '
            '(group_map callable), or (source_col + substrings)'
        )

    samples = samples.copy()

    if isinstance(group_map, dict):
        samples['group'] = samples.index.map(group_map)
    elif callable(group_map):
        samples['group'] = samples.apply(group_map, axis=1)
    else:
        if source_col not in samples.columns:
            raise ValueError(f'assign_groups: source_col {source_col!r} not in samples')
        ordered = list(substrings.items())

        def match(value):
            if pd.isna(value):
                return None
            s = str(value).lower()
            for label, needle in ordered:
                if needle.lower() in s:
                    return label
            return None

        samples['group'] = samples[source_col].map(match)

    counts = samples['group'].value_counts(dropna=False).to_dict()
    unassigned = counts.pop(np.nan, 0) if np.nan in counts else samples['group'].isna().sum()
    logger.info(
        'Assigned '
        + ', '.join(f'{v} {k}' for k, v in counts.items() if pd.notna(k))
        + (f', {unassigned} unassigned (dropped)' if unassigned else '')
    )
    return samples


'''
---SUBSET_BY_GROUP---
INPUT: expression (DataFrame), samples (DataFrame), groups (list of labels)
PROCESS: keep only samples whose 'group' is in the list
OUTPUT: (expression_subset, samples_subset)
'''
def subset_by_group(
    expression: pd.DataFrame,
    samples: pd.DataFrame,
    groups: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if 'group' not in samples.columns:
        raise ValueError("subset_by_group: samples has no 'group' column - call assign_groups first")
    mask = samples['group'].isin(groups)
    kept = samples.index[mask]
    missing = [gid for gid in kept if gid not in expression.columns]
    if missing:
        raise ValueError(f'subset_by_group: {len(missing)} sample(s) not in expression columns')
    return expression[kept].copy(), samples.loc[kept].copy()


'''
---NORMALIZE---
INPUT: expression (DataFrame), method ('log2' | 'quantile' | 'zscore')
PROCESS: apply the named normalization without mutating input
OUTPUT: new DataFrame, same shape/index/columns
'''
def normalize(expression: pd.DataFrame, method: str = 'log2') -> pd.DataFrame:
    if method == 'log2':
        # Not idempotent - caller must decide (see detect_log_scale).
        return np.log2(expression.clip(lower=0) + 1)

    if method == 'quantile':
        arr = expression.to_numpy(dtype=float)
        sorted_means = np.nanmean(np.sort(arr, axis=0), axis=1)
        ranks = expression.rank(method='average').to_numpy()
        n = arr.shape[0]
        positions = np.arange(1, n + 1)
        normalized = np.empty_like(arr)
        for j in range(arr.shape[1]):
            normalized[:, j] = np.interp(ranks[:, j], positions, sorted_means)
        return pd.DataFrame(normalized, index=expression.index, columns=expression.columns)

    if method == 'zscore':
        row_mean = expression.mean(axis=1)
        row_std = expression.std(axis=1).replace(0, np.nan)
        return expression.sub(row_mean, axis=0).div(row_std, axis=0)

    raise ValueError(
        f'Unknown normalize method: {method!r}. Expected one of: log2, quantile, zscore'
    )


'''
---DIFFERENTIAL_EXPRESSION---
INPUT: expression (log-scaled), samples, group_a, group_b, annotation (optional), group_col
PROCESS: Welch's t-test per gene, BH correction, merge gene_symbol if available
OUTPUT: DataFrame sorted by adj_p_value ascending
'''
def differential_expression(
    expression: pd.DataFrame,
    samples: pd.DataFrame,
    group_a: str,
    group_b: str,
    annotation: pd.DataFrame | None = None,
    group_col: str = 'group',
) -> pd.DataFrame:
    if group_col not in samples.columns:
        raise ValueError(f'differential_expression: samples has no {group_col!r} column')

    a_ids = samples.index[samples[group_col] == group_a]
    b_ids = samples.index[samples[group_col] == group_b]
    if len(a_ids) < 2 or len(b_ids) < 2:
        raise ValueError(
            f'differential_expression: need >=2 samples per group '
            f'(got {len(a_ids)} in {group_a!r}, {len(b_ids)} in {group_b!r})'
        )
    missing = [gid for gid in list(a_ids) + list(b_ids) if gid not in expression.columns]
    if missing:
        raise ValueError(
            f'differential_expression: {len(missing)} sample(s) not in expression columns'
        )

    a = expression[a_ids].to_numpy(dtype=float)
    b = expression[b_ids].to_numpy(dtype=float)

    t_stat, p_value = stats.ttest_ind(a, b, axis=1, equal_var=False, nan_policy='omit')
    t_stat = np.asarray(t_stat, dtype=float)
    p_value = np.asarray(p_value, dtype=float)

    mean_a = np.nanmean(a, axis=1)
    mean_b = np.nanmean(b, axis=1)
    log2FC = mean_a - mean_b

    adj_p_value = np.full_like(p_value, np.nan, dtype=float)
    valid = ~np.isnan(p_value)
    if valid.any():
        adj_p_value[valid] = multipletests(p_value[valid], method='fdr_bh')[1]

    de = pd.DataFrame({
        'probe_id': expression.index,
        'log2FC': log2FC,
        'mean_a': mean_a,
        'mean_b': mean_b,
        't_stat': t_stat,
        'p_value': p_value,
        'adj_p_value': adj_p_value,
    })

    if annotation is not None and 'gene_symbol' in annotation.columns:
        de = de.merge(
            annotation[['gene_symbol']],
            left_on='probe_id',
            right_index=True,
            how='left',
        )
        cols = ['probe_id', 'gene_symbol'] + [c for c in de.columns if c not in ('probe_id', 'gene_symbol')]
        de = de[cols]

    de = de.sort_values('adj_p_value', ascending=True, na_position='last').reset_index(drop=True)
    return de


'''
---TOP_DE_GENES---
INPUT: de_results, n, adj_p_max, abs_log2fc_min
PROCESS: filter by significance + fold-change, sort by |log2FC| desc
OUTPUT: top-N DataFrame
'''
def top_de_genes(
    de_results: pd.DataFrame,
    n: int = 50,
    adj_p_max: float = 0.05,
    abs_log2fc_min: float = 1.0,
) -> pd.DataFrame:
    filtered = de_results[
        (de_results['adj_p_value'] < adj_p_max)
        & (de_results['log2FC'].abs() >= abs_log2fc_min)
    ]
    if filtered.empty:
        logger.warning(
            f'top_de_genes: no genes pass adj_p<{adj_p_max} and |log2FC|>={abs_log2fc_min}'
        )
        return filtered
    ordered_idx = filtered['log2FC'].abs().sort_values(ascending=False).index
    return filtered.reindex(ordered_idx).head(n)
