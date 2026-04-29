"""Differential expression for a single GEO dataset.

Wraps shared_utils.differential_expression with log-scale auto-detection,
optional probe-to-symbol collapsing, and CSV emission. Called by ge.py;
not meant to be run directly.
"""

from __future__ import annotations
import logging
import os
import sys

import numpy as np
import pandas as pd

import shared_utils as su

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0,_ROOT)

logger = logging.getLogger(__name__)

_DE_CSV_NAME = 'de.csv'
_HEAD_N = 10

# public entry function

def run_diffex(
    expression: pd.DataFrame,
    samples: pd.DataFrame,
    annotation: pd.DataFrame | None,
    group_a: str,
    group_b: str,
    output_dir: str | None = None,
    *,
    group_col: str = 'group',
    collapse_to_gene: bool = False,
    print_head: bool = True,
) -> pd.DataFrame:
    """Run diff-expression and (optionally) write de.csv to output_dir.

    Returns the de_results DataFrame. If output_dir is given, also writes
    {output_dir}/de.csv. If print_head, writes the top-10 rows to stdout.
    """
    # validate group assignment
    if group_col not in samples.columns:
        raise ValueError(
            f"samples missing column {group_col!r}",
            f"Did you call shared_utils.assign_groups before run_diffex?",
            f"Available columns: {list(samples.columns)}"
        )
    # if matrix doesnt look log scaled, transform
    if su.detect_log_scale(expression):
        logger.info("expression detected as log-scale; skipping log2 transform")
        expr_log = expression
    else:
        logger.info("expression looks raw; applying log2 transform")
        expr_log = su.normalize(expression,'log2')

    # restrict to the 2 groups before testing
    expr_two, samp_two = su.subset_by_group(expr_log, samples, [group_a,group_b])
    logger.info(
        "diffex: %d genes x %d samples (%s=%d, %s=%d)",
        expr_two.shape[0], expr_two.shape[1],
        group_a, (samp_two[group_col] == group_a).sum(),
        group_b, (samp_two[group_col] == group_b).sum(),
    )

    # run t-test
    de = su.differential_expression(expr_two, samp_two, group_a, group_b, annotation=annotation, group_col=group_col)

    # collapse to gene -- keep row w smallest adj_p_value per gene_symbol
    if collapse_to_gene:
        if 'gene_symbol' not in de.columns:
            raise ValueError(
                "collapse_to_gene=True but de_results has no gene_symbol column. "
                "Pass an annotation frame to run_diffex."
            )
        before = len(de)
        de = (
            de.dropna(subset=['gene_symbol'])
                .sort_values('adj_p_value',na_position='last')
                .drop_duplicates(subset='gene_symbol', keep='first')
                .reset_index(drop=True)
        )
        logger.info('collapsed %d probes -> %d genes', before, len(de))
    # write to csv
    if output_dir is not None:
        os.makedirs(output_dir,exist_ok=True)
        csv_path = os.path.join(output_dir,_DE_CSV_NAME)
        de.to_csv(csv_path, index=False, float_format='%.6g')
        logger.info('wrote %s (%d rows)', csv_path, len(de))

    # print head for user viewing
    if print_head:
        cols = [c for c in ('probe_id','gene_symbol', 'log2FC','adj_p_value') if c in de.columns]
        print(de[cols].head(_HEAD_N).to_string(index=False))
    return de
