"""CLI orchestrator for the analyze_gene_expression project.

Subcommands: eda, diffex, volcano, heatmap, compare, all.
Each loads the GEO dataset(s) via shared_utils, dispatches to the
corresponding src/ module, and writes outputs under result/<accession>/.

Run `python ge.py --help` for usage. This file is the only intended
entry point — src/ modules expose Python functions, not __main__ blocks.
"""
from __future__ import annotations
import argparse
import logging
import os
import sys

import pandas as pd

import shared_utils as su
from src import eda, diffex, volcano, heatmap, compare

logger = logging.getLogger('ge')

_RESULT_ROOT = 'result'

# dispatch table
_HANDLERS = {
    'eda': _cmd_eda,
    'diffex': _cmd_diffex,
    'volcano': _cmd_volcano,
    'heatmap': _cmd_heatmap,
    'compare': _cmd_compare,
    'all': _cmd_all,
}

# Configure logger
def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

# path helpers
def _accession_dir(accession: str) -> str:
    # result/<accession>/ created on demand
    path = os.path.join(_RESULT_ROOT, accession)
    os.makedirs(path, exist_ok=True)
    return path

def _de_csv_path(accession: str, override: str | None=None) -> str:
    return override if override else os.path.join(_accession_dir(accession),'de.csv')

# argument parsing
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='ge',description='GEO gene-expression analysis CLI')
    p.add_argument('-v','--verbose',action='store_true',help='enable DEBUG logging')
    sub = p.add_subparsers(dest='command',required=True, metavar='COMMAND')

    # ----- EDA -----
    eda_p = sub.add_parser('eda',help='run EDA outputs for 1 dataset')
    eda_p.add_argument('--accession',required=True, help="GEO series accession (e.g. GSE19804) ")
    eda_p.add_argument('--group-col', help="Metadata column for group assignment (optional)")
    eda_p.add_argument('--group-a')
    eda_p.add_argument('--group-b')


    # ---- all ----
    all_p = sub.add_parser('all', help='run eda -> diffex -> volcano -> heatmap end-to-end')
    all_p.add_argument('--accession', required=True)
    all_p.add_argument('--group-col', required=True)
    all_p.add_argument('--group-a', required=True)
    all_p.add_argument('--group-b', required=True)
    all_p.add_argument('--top', type=int, default=50)

    return p

# shared loading & grouping helper for handlers
def _load_and_group(accession: str,
                     group_col: str | None, 
                     group_a: str | None,
                     group_b: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    # load via shared_utils and if all 3 groups given, assign groups
    expression, samples, annotation = su.load_geo_dataset(accession)
    if group_col and group_a and group_b:
        samples = su.assign_groups(
            samples,
            source_col=group_col,
            substrings={group_a: group_a, group_b: group_b}
        )
        counts = samples['group'].value_counts(dropna=False).to_dict()
        logger.info("group assignment for %s: %s", accession, counts)
    return expression, samples, annotation

# --------Handlers--------------
def _cmd_eda(args: argparse.Namespace) -> None:
    expression, samples, _ = _load_and_group(
        args.accession, args.group_col, args.group_a, args.group_b
    )
    out = os.path.join(_accession_dir(args.accession), 'eda')
    eda.run_eda(expression,samples, out)

def main(argv: list[str] | None=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    handler = _HANDLERS[args.command]
    handler(args)

if __name__=='__main__':
    main()