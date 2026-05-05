# Gene Expression Analysis from GEO — Implementation Plan

## Context

The user is starting a new data-science project at [analyze_gene_expression/](.) to analyze gene expression data from the NCBI Gene Expression Omnibus (GEO). The only existing artifact is [ge.py](ge.py) — an empty stub that the user has open in their IDE.

Required capabilities:
1. Load a GEO dataset by accession (e.g. `GSE12345`) and expose its expression matrix + sample metadata.
2. Run EDA on a single dataset — structure, missingness, sample distributions, PCA.
3. Normalize expression values.
4. Run differential expression between two sample groups (Welch's t-test + Benjamini–Hochberg FDR).
5. Produce a volcano plot (log2FC vs −log10 adjusted p).
6. Produce a heatmap of top differentially expressed genes with sample/gene clustering.
7. Compare two datasets — overlap of differentially expressed genes and fold-change concordance.

Conventions adopted from the two most mature projects in this repo ([../Covid-Test/](../Covid-Test/) and [../protein_visualization/](../protein_visualization/)):
- Modular scripts + a sibling `shared_utils.py` for reusable helpers.
- `src/` folder with one script per output/plot type (matches [../Covid-Test/src/](../Covid-Test/src/)).
- `data/` for input caches, `result/` for output artifacts (matches [../Covid-Test/data/](../Covid-Test/data/), [../Covid-Test/result/](../Covid-Test/result/)).
- Pinned `requirements.txt` (matches [../protein_visualization/requirements.txt](../protein_visualization/requirements.txt), which already lists `GEOparse`).
- Detailed plan doc checked in alongside code (matches [../Covid-Test/covid-test-plan.md](../Covid-Test/covid-test-plan.md) and [../protein_visualization/protein_visualization_plan.md](../protein_visualization/protein_visualization_plan.md)).

User-selected choices (from clarifying questions):
- **Architecture:** modular scripts + `shared_utils.py`.
- **Data input:** auto-download by GEO accession via `GEOparse`, with local cache.
- **Plotting:** matplotlib + seaborn, static PNG output.
- **Stats:** Welch's t-test with Benjamini–Hochberg FDR (only — no Mann–Whitney or limma in v1).

## File structure

Create under [analyze_gene_expression/](.):

```
analyze_gene_expression/
  ge.py                       # CLI orchestrator (fill the empty stub)
  shared_utils.py             # GEO loading, normalization, diff-expression
  requirements.txt            # pinned deps
  gene_expression_plan.md     # this plan
  readme.md                   # usage docs
  data/                       # GEOparse SOFT cache (gitignored, created at runtime)
  result/                     # PNG outputs (created at runtime)
  src/
    eda.py                    # EDA plots for one dataset
    diffex.py                 # run diff-expression and write results CSV
    volcano.py                # volcano plot
    heatmap.py                # clustered heatmap of top DE genes
    compare.py                # compare two datasets
```

No Jupyter notebooks — matches repo-wide convention (all 16 existing projects are `.py` files).

## Dependencies — `requirements.txt`

```
GEOparse>=2.0
pandas>=2.0
numpy>=1.24
scipy>=1.10
statsmodels>=0.14
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.13
```

Notes:
- `GEOparse` is already vetted in [../protein_visualization/requirements.txt](../protein_visualization/requirements.txt).
- `statsmodels` is needed for `multipletests(..., method='fdr_bh')`; `scipy` does not ship BH correction.
- `scikit-learn` is used only for PCA and `StandardScaler` in EDA.
- No Dash / plotly — plots are static PNGs written to `result/`.

## Core data model

Throughout the project, the standard in-memory objects are:

| Object              | Shape / type                                 | Notes                                                              |
|---------------------|-----------------------------------------------|--------------------------------------------------------------------|
| `expression`        | `pd.DataFrame`, rows = probes/genes, cols = samples (GSM IDs) | Values are raw expression from the SOFT file (often log-scale already). |
| `samples`           | `pd.DataFrame`, rows = GSM IDs               | Metadata columns harvested from `GSM.metadata` — `title`, `source_name_ch1`, `characteristics_ch1`, plus a user-assigned `group` column. |
| `annotation`        | `pd.DataFrame`, rows = probe IDs             | Platform annotation merged from `GPL.table`, giving `gene_symbol`, `gene_title`, etc. Optional — only used for labeling. |
| `de_results`        | `pd.DataFrame`                               | Columns: `probe_id`, `gene_symbol`, `log2FC`, `mean_a`, `mean_b`, `t_stat`, `p_value`, `adj_p_value`. |

Gene orientation is **rows = genes, columns = samples** throughout (bioinformatics convention; matches GEOparse's `pivot_samples('VALUE')` output).

## `shared_utils.py` — helper signatures

### Module-level structure

```python
"""Shared helpers for GEO gene-expression analysis.

Kept deliberately narrow: loading, group assignment, normalization, diff-
expression, top-gene selection, plus a couple of small utilities
(`detect_log_scale`, `subset_by_group`) reused across `src/` modules.

All functions are pure — no file I/O except `load_geo_dataset` (which caches
to disk) and no mutation of inputs (all transforms return new frames).
"""

from __future__ import annotations
import logging
import os
import warnings
from typing import Callable

import numpy as np
import pandas as pd
import GEOparse
from scipy import stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)
```

Module conventions:
- Every public function takes DataFrames by value and returns **new** DataFrames — never mutate inputs in place. Downstream modules rely on being able to re-use the same `expression` object across EDA, normalization, and diff-expression calls.
- Log via `logger.info` / `logger.warning`, not `print`. The CLI (`ge.py`) configures the root logger once; library code stays quiet unless something is worth saying.
- Raise `ValueError` for user-facing precondition failures (bad group name, empty matrix) and let GEOparse/pandas raise their own exceptions for lower-level problems after prepending accession context.

### Full signatures

```python
def load_geo_dataset(
    accession: str,
    cache_dir: str = 'data',
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download (if needed) and parse a GEO series. Return (expression, samples, annotation)."""

def detect_log_scale(expression: pd.DataFrame) -> bool:
    """Heuristic: GEO microarray data is typically already log2-transformed when
    the global median is < 30 and the max is < 100. Raw intensities are usually
    in the thousands. Returns True if the matrix looks log-scaled."""

def assign_groups(
    samples: pd.DataFrame,
    group_map: dict[str, str] | Callable[[pd.Series], str | None],
    source_col: str | None = None,
    substrings: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Add a 'group' column to samples metadata. Three calling styles:

    1. Explicit dict: {GSM_id: group_label}.
    2. Callable: fn(row) -> label or None (None drops the sample).
    3. Substring match: source_col='source_name_ch1',
       substrings={'tumor': 'tumor', 'normal': 'normal'} — first match wins."""

def subset_by_group(
    expression: pd.DataFrame,
    samples: pd.DataFrame,
    groups: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter expression + samples down to rows/columns in the named groups.
    Used by diffex and heatmap to drop unassigned samples consistently."""

def normalize(
    expression: pd.DataFrame,
    method: str = 'log2',
) -> pd.DataFrame:
    """Return a normalized copy. Methods: 'log2' (log2(x+1)), 'quantile', 'zscore'.
    Preserves index/columns. Does NOT mutate the input."""

def differential_expression(
    expression: pd.DataFrame,
    samples: pd.DataFrame,
    group_a: str,
    group_b: str,
    annotation: pd.DataFrame | None = None,
    group_col: str = 'group',
) -> pd.DataFrame:
    """Welch's t-test per gene between samples[group_col]==group_a and ==group_b.
    Adds BH-corrected adj_p_value via statsmodels.multipletests(method='fdr_bh').
    log2FC = mean_a - mean_b (inputs assumed already log-scaled).
    If annotation is given, merges gene_symbol onto the result."""

def top_de_genes(
    de_results: pd.DataFrame,
    n: int = 50,
    adj_p_max: float = 0.05,
    abs_log2fc_min: float = 1.0,
) -> pd.DataFrame:
    """Return top-N rows sorted by |log2FC| among genes meeting
    adj_p_value < adj_p_max AND |log2FC| >= abs_log2fc_min."""
```

### `load_geo_dataset` — details

1. `os.makedirs(cache_dir, exist_ok=True)`.
2. `gse = GEOparse.get_GEO(geo=accession, destdir=cache_dir, silent=True)` — downloads `.soft.gz` on first call, reads from cache thereafter. Wrap in try/except and re-raise with `f"Failed to load {accession}: {original_exception}"` so the caller immediately sees which accession failed.
3. Build **expression matrix**:
   - `expression = gse.pivot_samples('VALUE')` — returns DataFrame indexed by probe ID (`ID_REF`), columns = GSM IDs.
   - `expression = expression.apply(pd.to_numeric, errors='coerce')` — some SOFT files ship numeric columns as strings.
   - `expression = expression.dropna(how='all')` — drop probes that are NaN across every sample.
   - If `expression.empty` after the drop, raise `ValueError(f"{accession}: expression matrix is empty after NaN filtering")`.
4. Build **samples DataFrame**:
   - For each `gsm_id, gsm in gse.gsms.items()`, flatten `gsm.metadata` — each value is a list, join with `' | '` so the cell remains scalar.
   - Index by GSM ID (set `index.name = 'gsm_id'`).
   - Reorder columns so commonly-useful ones come first: `['title', 'source_name_ch1', 'characteristics_ch1', 'platform_id']` then the rest alphabetically. Columns that don't exist on every GSM should be filled with NaN (use `pd.DataFrame.from_records` with a consistent key set).
5. Build **annotation DataFrame**:
   - `platforms = list(gse.gpls.values())`. If `len(platforms) > 1`, `logger.warning(f"{accession} has {len(platforms)} platforms; using {platforms[0].name}")` and use the first.
   - `annotation = platforms[0].table.copy()`.
   - If `'ID'` column exists, `annotation = annotation.set_index('ID')`; otherwise fall back to the first column and log a warning.
   - Rename for consistency: `{'Gene Symbol': 'gene_symbol', 'Gene Title': 'gene_title', 'ENTREZ_GENE_ID': 'entrez_id'}` — only rename keys that exist (use `.rename(columns=mapping)` which silently ignores missing keys).
   - If no `gene_symbol` column ended up in annotation after renaming, `logger.warning` — downstream plots will fall back to probe IDs.

Failure modes to handle explicitly:
- **Accession not found / network error**: re-raise with accession prepended.
- **Multi-platform series**: warn + use first platform (documented in the docstring as a v1 limitation).
- **Empty expression matrix** after NaN drop: raise `ValueError`.
- **GEOparse cache file corrupted** (happens if a download was interrupted): catch the parse error, delete the `.soft.gz`, retry once. If it fails again, re-raise.

### `detect_log_scale` — details

```python
def detect_log_scale(expression: pd.DataFrame) -> bool:
    median = np.nanmedian(expression.values)
    maximum = np.nanmax(expression.values)
    return median < 30 and maximum < 100
```

Used by `src/diffex.py` to decide whether to `normalize(..., 'log2')` before the t-test. GEO microarray data is often already log-transformed (values 2–15), but raw-intensity series still show up and would give nonsensical log2FC if left untransformed.

### `assign_groups` — details

Three branches — exactly one of `group_map`, `source_col + substrings`, or a callable `group_map` must be supplied.

1. **Dict form** (`group_map: dict[str, str]`):
   - `samples['group'] = samples.index.map(group_map)` — missing keys become NaN.
2. **Callable form** (`group_map: Callable[[pd.Series], str | None]`):
   - `samples['group'] = samples.apply(group_map, axis=1)` — callable returns `None` to drop.
3. **Substring form** (`source_col='characteristics_ch1'`, `substrings={'tumor': 'tumor', 'normal': 'normal'}`):
   - For each row, lowercase `samples.loc[idx, source_col]` and check each substring (also lowercased) in insertion order — first match wins.
   - Ties go to the first key in the dict; document this.

Always return a **copy** (`samples = samples.copy()`) — caller's frame is untouched. Log counts per group at `INFO` level so the CLI prints `"Assigned 58 tumor, 60 normal, 2 unassigned (dropped)"`.

### `subset_by_group` — details

```python
def subset_by_group(expression, samples, groups):
    mask = samples['group'].isin(groups)
    kept = samples.index[mask]
    return expression[kept], samples.loc[kept]
```

Used wherever downstream code needs "only samples in group A or B" — diffex, heatmap, PCA coloring. Keeping it as a shared helper means the group-dropping rule (and what happens to probes with all-NaN in the subset) stays consistent.

### `normalize` — details

- **`log2`**: `np.log2(df.clip(lower=0) + 1)`. Clip guards against negative values (some background-subtracted arrays produce them). Docstring warns: **not idempotent** — calling twice double-logs. `src/diffex.py` uses `detect_log_scale` first to avoid this.
- **`quantile`**: Implementation (no extra dependency):
  ```python
  ranks = df.rank(method='average')
  sorted_means = np.sort(df.values, axis=0).mean(axis=1)  # mean across samples at each rank
  # Map each rank → the sorted-mean at that rank position
  normalized = ranks.apply(lambda col: np.interp(col, np.arange(1, len(col)+1), sorted_means))
  ```
  Makes every sample share the same distribution — removes technical variation between arrays. Docstring notes: destroys absolute scale, only use before comparisons that are scale-invariant.
- **`zscore`**: per-gene (per-row) z-score:
  ```python
  (df.sub(df.mean(axis=1), axis=0)).div(df.std(axis=1).replace(0, np.nan), axis=0)
  ```
  `std==0` rows become NaN rather than `inf` — heatmap code drops them. Used specifically for heatmap row scaling, **not** before diff-expression (which needs raw log-scale means for log2FC).

Unknown `method` → `ValueError(f"Unknown normalize method: {method!r}. Expected one of: log2, quantile, zscore")`.

### `differential_expression` — details

Preconditions checked up front (raise `ValueError` on violation):
- `group_col` exists in `samples.columns`.
- At least 2 samples where `samples[group_col] == group_a`, same for `group_b` (t-test needs n≥2 per group and Welch's breaks on n=1).
- Expression columns must be a superset of the filtered sample IDs (no dangling references).

Algorithm:
1. `a_ids = samples.index[samples[group_col] == group_a]`; same for `b_ids`.
2. `a = expression[a_ids].to_numpy(dtype=float)`, `b = expression[b_ids].to_numpy(dtype=float)` — shape `(n_genes, n_a)` and `(n_genes, n_b)`.
3. **Vectorized t-test**: `t_stat, p_value = scipy.stats.ttest_ind(a, b, axis=1, equal_var=False, nan_policy='omit')` — one call for all genes. Welch's t-test does not assume equal variance.
4. `mean_a = np.nanmean(a, axis=1)`, `mean_b = np.nanmean(b, axis=1)`, `log2FC = mean_a - mean_b` (inputs are log-scale, so subtraction of log-means is log2FC).
5. **Handle NaN p-values**: genes with all-NaN or constant values in one group return NaN from `ttest_ind`. For BH correction, statsmodels' `multipletests` rejects NaN input — mask them out, correct the rest, re-insert NaN:
   ```python
   adj_p_value = np.full_like(p_value, np.nan)
   valid = ~np.isnan(p_value)
   adj_p_value[valid] = multipletests(p_value[valid], method='fdr_bh')[1]
   ```
6. Assemble DataFrame:
   ```python
   de = pd.DataFrame({
       'probe_id': expression.index,
       'log2FC': log2FC,
       'mean_a': mean_a,
       'mean_b': mean_b,
       't_stat': t_stat,
       'p_value': p_value,
       'adj_p_value': adj_p_value,
   })
   ```
7. If `annotation is not None` and has `gene_symbol`, left-join on probe ID:
   `de = de.merge(annotation[['gene_symbol']], left_on='probe_id', right_index=True, how='left')`.
   Reorder so `gene_symbol` comes right after `probe_id`.
8. Sort by `adj_p_value` ascending (NaNs last), reset index, return.

Docstring should state explicitly: **inputs assumed log-scaled**. If caller forgets and passes raw intensities, `log2FC` will be nonsense (a ratio interpreted as a log difference).

### `top_de_genes` — details

```python
def top_de_genes(de_results, n=50, adj_p_max=0.05, abs_log2fc_min=1.0):
    filtered = de_results[
        (de_results['adj_p_value'] < adj_p_max)
        & (de_results['log2FC'].abs() >= abs_log2fc_min)
    ]
    return filtered.reindex(filtered['log2FC'].abs().sort_values(ascending=False).index).head(n)
```

- Sort by `|log2FC|` descending (largest effect size first), not by p-value — when hundreds of genes clear the significance threshold, effect size is the more useful tiebreaker for plotting.
- If fewer than `n` genes pass the thresholds, return however many there are. Log a warning if the result is empty — typically means the caller picked thresholds too strict for their data.
- Input is expected to be the frame returned by `differential_expression` — relies on columns `adj_p_value` and `log2FC`.

## `ge.py` — CLI orchestrator

Single entry point that calls the individual `src/` modules. Uses `argparse` with subcommands. Writes outputs to `result/<accession>/` so runs don't collide.

```
python ge.py eda      --accession GSE19804
python ge.py diffex   --accession GSE19804 --group-col 'source_name_ch1' --group-a 'tumor' --group-b 'normal'
python ge.py volcano  --accession GSE19804 --from-results result/GSE19804/de.csv
python ge.py heatmap  --accession GSE19804 --from-results result/GSE19804/de.csv --top 50
python ge.py compare  --accession-a GSE19804 --accession-b GSE10072 --group-a tumor --group-b normal
```

Each subcommand:
- Loads the dataset(s) via `shared_utils.load_geo_dataset`.
- Calls the corresponding module in `src/`.
- Writes PNG/CSV outputs into `result/<accession>/`.

A `python ge.py all --accession ... --group-a ... --group-b ...` convenience subcommand runs eda → diffex → volcano → heatmap end-to-end.

Group assignment at the CLI: user supplies `--group-col` (metadata column name) plus `--group-a` / `--group-b` substrings. `shared_utils.assign_groups` maps each sample: if the substring appears in the metadata value → that group; samples matching neither are dropped before diffex.

## `src/eda.py` — EDA outputs

Inputs: `expression`, `samples`, output directory. Produces one text summary and four PNG plots under `result/<accession>/eda/`. This is the first matplotlib/seaborn module in the repo, so it also sets the style baseline (dpi, savefig args, figure-close discipline) for [src/volcano.py](src/volcano.py) and [src/heatmap.py](src/heatmap.py) later.

### Design decisions

- **Callable entry, not a procedural script.** [ge.py](ge.py) is the CLI orchestrator that loads data and calls into `src/` modules (see "`ge.py` — CLI orchestrator" section above). So `eda.py` exports `run_eda(expression, samples, output_dir)` rather than running top-to-bottom like [../Covid-Test/src/covid.py](../Covid-Test/src/covid.py). Keeps `ge.py` responsible for I/O and lets `run_eda` be reusable from the `all` subcommand.
- **One private helper per output.** Five outputs, five `_write_*` / `_plot_*` helpers called in sequence. Easier to debug a single broken plot than a 200-line `run_eda`.
- **Logger, not print.** Matches [shared_utils.py](shared_utils.py) convention. The CLI configures the root logger; `src/` modules stay quiet unless they have something useful to say.
- **`samples['group']` is optional.** EDA may run before group assignment. Handle missing column by falling back to no-color / no-grouping rather than raising.
- **`plt.close(fig)` after every save.** Without this, repeated CLI runs leak figures and seaborn's global style state bleeds between plots.

### Module header

```python
"""Exploratory data analysis for a single GEO dataset.

Writes one text summary plus four PNG plots to result/<accession>/eda/.
Called by ge.py; not meant to be run directly. The caller is expected to
have already invoked shared_utils.load_geo_dataset and (optionally)
assign_groups before passing expression and samples in.
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
```

The `_ROOT` block matches the existing Covid-Test pattern for `src/` scripts importing from their parent. `_SAVEFIG_KW` is a single source of truth for savefig args used in every plot helper. Note: only `detect_log_scale` is imported from `shared_utils` — do **not** call `normalize` here. EDA reports on the raw matrix as loaded; normalization is `diffex.py`'s concern.

### Public entry function

```python
def run_eda(expression: pd.DataFrame,
            samples: pd.DataFrame,
            output_dir: str) -> None:
    """Run all EDA outputs into output_dir (created if needed)."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info("EDA on %d genes x %d samples -> %s",
                expression.shape[0], expression.shape[1], output_dir)
    _write_summary(expression, samples, output_dir)
    _plot_sample_boxplots(expression, output_dir)
    _plot_correlation_heatmap(expression, samples, output_dir)
    _plot_pca(expression, samples, output_dir)
    _plot_gene_variance(expression, output_dir)
```

Five sequential calls, no try/except — let any plot failure abort and surface the traceback. EDA is fast and idempotent; partial output is worse than a clean failure.

### `_write_summary` — do this first; pure-text, easy to verify

Pure-text output, no plotting — implement before any of the `_plot_*` helpers so the rest of `run_eda` has a known-good template to copy (logger pattern, output path, file handling).

#### Signature

```python
def _write_summary(expression: pd.DataFrame,
                   samples: pd.DataFrame,
                   output_dir: str) -> None:
    """Write summary.txt with shape, value range, NaN stats, and group counts."""
```

No return value, no exceptions caught — let pandas/numpy raise on bad input. Caller (`run_eda`) has already done `os.makedirs(output_dir, exist_ok=True)`, so this helper does not re-create the directory.

#### Step-by-step implementation

1. **Compute scalars up front** so the file-writing block is dumb formatting only. Use `expression.values` (a 2-D numpy array) for the global stats — `np.nanmin`/`np.nanmax`/`np.nanmedian` on the DataFrame work but emit a `FutureWarning` in pandas ≥ 2.1.

   ```python
   n_genes, n_samples = expression.shape
   values = expression.values
   value_min = float(np.nanmin(values))
   value_max = float(np.nanmax(values))
   value_median = float(np.nanmedian(values))
   log_scale = detect_log_scale(expression)
   ```

   Cast to `float` — `np.nanmin` returns a numpy scalar; formatting a numpy float with `{:.3f}` works but the cast keeps the line type-stable for any future JSON dump.

2. **Per-sample NaN fraction** — one fraction per column, then summarize with min/median/max:

   ```python
   nan_frac = expression.isna().mean(axis=0)  # Series, one entry per sample
   nan_min = float(nan_frac.min())
   nan_median = float(nan_frac.median())
   nan_max = float(nan_frac.max())
   ```

   `axis=0` means "average across rows for each column" — the column is a sample, so this gives the fraction of probes that are NaN within each sample. `.min()`/`.max()` on an empty Series returns NaN; that only happens if `expression` has zero columns, which `load_geo_dataset` already rejects.

3. **Group counts (optional)** — only if the caller assigned groups before calling `run_eda`:

   ```python
   if 'group' in samples.columns:
       group_counts = samples['group'].value_counts(dropna=False).to_dict()
   else:
       group_counts = None
   ```

   `dropna=False` so unassigned samples (NaN in the group column) appear as a `nan` key — useful for spotting "I forgot to call assign_groups" without having to read the boxplot.

4. **Build the output as a list of strings**, then join with newlines. Easier to read in code review than seven `f.write(...)` calls and trivially testable (you can assert against the list before writing).

   ```python
   lines = [
       f"Genes (rows): {n_genes}",
       f"Samples (cols): {n_samples}",
       f"Value range: [{value_min:.3f}, {value_max:.3f}]",
       f"Global median: {value_median:.3f}",
       f"Detected log-scale: {log_scale}",
       f"Per-sample NaN fraction: min={nan_min:.4f}, median={nan_median:.4f}, max={nan_max:.4f}",
   ]
   if group_counts is not None:
       lines.append(f"Group counts: {group_counts}")
   else:
       lines.append("Group counts: <not assigned>")
   ```

   Always emit the seventh line — even when groups are unassigned — so downstream verification can count exactly seven lines without branching.

5. **Write atomically-ish**: open with a context manager, trailing newline so the file is POSIX-correct.

   ```python
   path = os.path.join(output_dir, 'summary.txt')
   with open(path, 'w', encoding='utf-8') as f:
       f.write('\n'.join(lines) + '\n')
   logger.info("wrote %s (%d genes x %d samples)", path, n_genes, n_samples)
   ```

   `encoding='utf-8'` is explicit because Windows defaults to cp1252 and group labels could contain non-ASCII characters from `characteristics_ch1`.

#### Edge cases to be aware of

- **All-NaN matrix** → `np.nanmin` raises a `RuntimeWarning` and returns NaN. `load_geo_dataset` already raises `ValueError` on an empty matrix after `dropna(how='all')`, but a matrix where every cell is NaN within at least one row would still slip through. Acceptable for v1 — the warning is loud enough; don't add code to suppress it.
- **`samples` row order vs `expression` columns** — this helper does not rely on alignment. It reads `samples['group']` independently of `expression`. Don't add a sanity check; that belongs in `differential_expression`.
- **Long group labels** — `value_counts().to_dict()` produces a `dict` whose `repr` can wrap awkwardly in a text editor. Acceptable; the file is for humans glancing at it, not parsing.

#### Verification

After running [test.py](test.py) (which calls `run_eda`), `result/GSE19804/eda/summary.txt` should contain exactly 7 lines. Concrete expectation for GSE19804 (Affymetrix HG-U133 Plus 2.0, log-scaled):

```
Genes (rows): 54675
Samples (cols): 120
Value range: [1.xxx, 14.xxx]
Global median: 6.xxx
Detected log-scale: True
Per-sample NaN fraction: min=0.0000, median=0.0000, max=0.0xxx
Group counts: {'tumor': 60, 'normal': 60}
```

If `Detected log-scale` is `False` here, something went wrong upstream (probably the SOFT file's `VALUE` column is raw intensity for this series — investigate before proceeding to diffex). If `Group counts` shows `<not assigned>`, the caller forgot `assign_groups` — re-run after fixing.

### `_plot_sample_boxplots` — details

```python
def _plot_sample_boxplots(expression, output_dir):
    n = expression.shape[1]
    sub = expression.iloc[:, :_MAX_BOXPLOT_SAMPLES] if n > _MAX_BOXPLOT_SAMPLES else expression
    fig, ax = plt.subplots(figsize=(max(6, sub.shape[1] * 0.25), 5))
    sns.boxplot(data=sub, ax=ax, fliersize=1, linewidth=0.5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.set_ylabel('Expression')
    ax.set_title(f'Per-sample expression distribution (first {sub.shape[1]} of {n})')
    fig.savefig(os.path.join(output_dir, 'sample_boxplots.png'), **_SAVEFIG_KW)
    plt.close(fig)
```

Subset to the first 40 samples — the actual plot is unreadable above ~50 boxes. Width scales with sample count so labels don't crush.

### `_plot_correlation_heatmap` — details

```python
def _plot_correlation_heatmap(expression, samples, output_dir):
    corr = expression.corr()  # sample-sample correlation
    if 'group' in samples.columns:
        order = samples.dropna(subset=['group']).sort_values('group').index
        order = [s for s in order if s in corr.index]
        corr = corr.loc[order, order]
    fig, ax = plt.subplots(figsize=(max(6, corr.shape[0] * 0.15),
                                    max(5, corr.shape[0] * 0.15)))
    sns.heatmap(corr, cmap='viridis', square=True, ax=ax,
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'Pearson r'})
    ax.set_title('Sample-sample correlation')
    fig.savefig(os.path.join(output_dir, 'sample_correlation_heatmap.png'), **_SAVEFIG_KW)
    plt.close(fig)
```

`expression.corr()` operates column-wise — perfect for sample-sample. Hide tick labels (they'd overlap into a smear above ~20 samples). Sorting by group makes the block-diagonal structure visible without needing colored side-bars.

### `_plot_pca` — details

```python
def _plot_pca(expression, samples, output_dir):
    mat = expression.dropna(axis=0, how='any')
    if mat.empty:
        logger.warning("PCA skipped: no genes without NaN")
        return
    X = StandardScaler().fit_transform(mat.T.values)  # samples as rows
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    if 'group' in samples.columns:
        for grp, sub in samples.groupby('group'):
            idx = [samples.index.get_loc(s) for s in sub.index if s in samples.index]
            ax.scatter(coords[idx, 0], coords[idx, 1], label=str(grp), s=30, alpha=0.8)
        ax.legend(title='group')
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=30, alpha=0.8)
    ax.set_xlabel(f'PC1 ({var[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({var[1]:.1f}%)')
    ax.set_title('PCA of samples')
    fig.savefig(os.path.join(output_dir, 'pca_scatter.png'), **_SAVEFIG_KW)
    plt.close(fig)
```

Three easy-to-miss details:
1. **Transpose before PCA** — `expression.T` puts samples as rows (one observation per sample). Forgetting this is the most common bug; you'll get gene-PCA, not sample-PCA.
2. **`StandardScaler()` first** — without it, high-variance genes dominate PC1 and the projection is dominated by a few outliers.
3. **Drop NaN rows** — `sklearn.PCA` raises on NaN. Even after `load_geo_dataset`'s `dropna(how='all')` there may be rows with partial NaN.

Color-by-group re-locates each sample's row in `samples.index` to index into `coords` — `samples` row order may not match `mat.T` row order.

### `_plot_gene_variance` — details

```python
def _plot_gene_variance(expression, output_dir):
    var = expression.var(axis=1).dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(var, bins=80)
    ax.set_xlabel('Per-gene variance')
    ax.set_ylabel('Gene count')
    ax.set_yscale('log')
    ax.set_title('Per-gene variance distribution')
    fig.savefig(os.path.join(output_dir, 'gene_variance_hist.png'), **_SAVEFIG_KW)
    plt.close(fig)
```

`yscale='log'` is important — variance histograms are heavily right-skewed; on linear scale the long tail of high-variance genes (the interesting ones) is invisible.

No seaborn `clustermap` here — that lives in [src/heatmap.py](src/heatmap.py).

### Verification for `eda.py`

Extend [test.py](test.py) with:

```python
import logging, os
from src import eda

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
out = os.path.join('result', 'GSE19804', 'eda')
eda.run_eda(exp, samp, out)
print('outputs:', sorted(os.listdir(out)))
```

Expected:
1. Five files appear in `result/GSE19804/eda/`: `summary.txt`, `sample_boxplots.png`, `sample_correlation_heatmap.png`, `pca_scatter.png`, `gene_variance_hist.png`.
2. `summary.txt` reports ~54k genes × 120 samples, detected log-scale = `True`, group counts ~60 tumor / 60 normal.
3. `pca_scatter.png` — tumor vs normal samples should separate visibly along PC1 or PC2 (strong lung-cancer signal in GSE19804).
4. `sample_correlation_heatmap.png` — visible 2-block structure when sorted by group.
5. Re-run; outputs are overwritten without error (idempotency check).

If PC separation is weak, sanity-check that `samp['group']` was actually populated before calling `run_eda` — passing the un-grouped `sample` frame is the most likely bug.

## `src/diffex.py` — differential expression

Wraps `shared_utils.differential_expression` with the boilerplate that every caller would otherwise duplicate: log-scale detection, group assignment, optional probe-to-symbol collapsing, CSV emission, and a printable head. The actual t-test math lives in [shared_utils.py](shared_utils.py) — this module is glue, not statistics.

### Design decisions

- **Callable entry, not a procedural script.** Mirrors [src/eda.py](src/eda.py): `ge.py` owns I/O and dispatches to a function. `run_diffex(expression, samples, annotation, group_a, group_b, output_dir)` is the shape, returning the `de_results` DataFrame so the `all` orchestrator in `ge.py` can hand it straight to volcano/heatmap without a CSV round-trip.
- **Log-scale detection happens here, not in `ge.py`.** The CLI doesn't need to know that diffex requires log-scaled inputs — that's a concern of the diffex pipeline. `ge.py`'s `_cmd_diffex` currently runs `detect_log_scale` itself; move that responsibility into `run_diffex` and simplify the handler.
- **Probe collapsing is opt-in.** Some downstream uses (volcano, heatmap) want one row per probe; others (`compare`) need one row per `gene_symbol`. Provide a `collapse_to_gene` flag, default `False` — diffex emits the full probe-level table by default, callers ask for collapse explicitly.
- **CSV is a side effect, not the return value.** Returning the DataFrame keeps the function composable; writing the CSV when `output_dir` is given keeps the CLI ergonomics intact. Two responsibilities, one function — acceptable because they're inseparable in practice (every caller wants both).
- **No try/except.** Same convention as `eda.py` — let `differential_expression`'s `ValueError`s propagate. The user runs from the CLI; a stack trace is more useful than a swallowed error.

### Module header

```python
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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import shared_utils as su

logger = logging.getLogger(__name__)

_DE_CSV_NAME = 'de.csv'
_HEAD_N = 10
```

The `_ROOT`/sys.path block matches [src/eda.py](src/eda.py) — keep the pattern identical so future `src/` modules can copy it. Importing `shared_utils as su` (not `from shared_utils import ...`) makes the call sites `su.differential_expression(...)` — easier to grep for and matches [test.py](test.py)'s convention.

### Public entry function

```python
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
```

The keyword-only divider (`*`) forces callers to spell out the optional knobs — `collapse_to_gene=True` is the kind of flag that should never be a positional surprise. `output_dir` is positional-but-optional so the typical CLI invocation (`run_diffex(exp, samp, ann, 'tumor', 'normal', out_dir)`) reads naturally.

### Step-by-step implementation

1. **Validate group assignment up front.** `shared_utils.differential_expression` will raise its own `ValueError` if `group_col` is missing, but the message is opaque ("group_col 'group' not in samples"). Catch the missing-group-column case here with a more actionable error — diffex is the typical first place users notice they forgot `assign_groups`:

   ```python
   if group_col not in samples.columns:
       raise ValueError(
           f"samples missing column {group_col!r}. "
           f"Did you call shared_utils.assign_groups before run_diffex? "
           f"Available columns: {list(samples.columns)}"
       )
   ```

   Don't validate `group_a` / `group_b` membership — let `differential_expression` raise its "fewer than 2 samples in group X" error, which already includes the count.

2. **Auto-log if needed.** Reuse the heuristic from EDA — if the matrix doesn't look log-scaled, transform once. Log it loudly so the user can spot misclassification:

   ```python
   if su.detect_log_scale(expression):
       logger.info("expression detected as log-scale; skipping log2 transform")
       expr_log = expression
   else:
       logger.info("expression looks raw-intensity; applying log2 transform")
       expr_log = su.normalize(expression, 'log2')
   ```

   Bind to a new name (`expr_log`) — don't reassign `expression`. Keeping the original around is cheap and means a downstream bug ("why is `expression` modified?") can't blame this function. `shared_utils.normalize` already returns a copy, so this is correctness-via-naming, not via copying.

3. **Restrict to the two groups before testing.** `differential_expression` filters internally, but doing it here means the log message at step 2 reflected the matrix actually used and any `print_head` output below references the right shape:

   ```python
   expr_two, samp_two = su.subset_by_group(expr_log, samples, [group_a, group_b])
   logger.info(
       "diffex: %d genes x %d samples (%s=%d, %s=%d)",
       expr_two.shape[0], expr_two.shape[1],
       group_a, (samp_two[group_col] == group_a).sum(),
       group_b, (samp_two[group_col] == group_b).sum(),
   )
   ```

   `subset_by_group` is shared with heatmap; using it here keeps the "drop unassigned samples" rule centralized.

4. **Run the t-test.** One call into `shared_utils`:

   ```python
   de = su.differential_expression(
       expr_two, samp_two, group_a, group_b,
       annotation=annotation, group_col=group_col,
   )
   ```

   The returned frame is sorted by `adj_p_value` ascending with NaNs last (per the `differential_expression` spec). Don't re-sort.

5. **Optional probe-to-gene collapse.** Only if requested. Strategy: keep the row with the smallest `adj_p_value` per `gene_symbol`. NaN symbols are dropped (probes that never mapped to a gene aren't useful for cross-platform comparison):

   ```python
   if collapse_to_gene:
       if 'gene_symbol' not in de.columns:
           raise ValueError(
               "collapse_to_gene=True but de_results has no gene_symbol column. "
               "Pass an annotation frame to run_diffex."
           )
       before = len(de)
       de = (
           de.dropna(subset=['gene_symbol'])
             .sort_values('adj_p_value', na_position='last')
             .drop_duplicates(subset='gene_symbol', keep='first')
             .reset_index(drop=True)
       )
       logger.info("collapsed %d probes -> %d genes", before, len(de))
   ```

   `keep='first'` after sort-by-`adj_p_value` is the "smallest p wins" rule. `na_position='last'` makes NaN p-values lose to any real p-value during the dedupe.

6. **Write CSV (when `output_dir` is given).** `os.makedirs` with `exist_ok=True` so callers don't have to pre-create the directory. Float format keeps the file readable in a text editor without losing precision for the tail-end small p-values:

   ```python
   if output_dir is not None:
       os.makedirs(output_dir, exist_ok=True)
       csv_path = os.path.join(output_dir, _DE_CSV_NAME)
       de.to_csv(csv_path, index=False, float_format='%.6g')
       logger.info("wrote %s (%d rows)", csv_path, len(de))
   ```

   `index=False` because the index after `differential_expression` is just `0..N-1` — meaningful information lives in the columns. `'%.6g'` keeps `1.234e-15`-style scientific notation for tiny p-values; `'%.6f'` would round them all to zero.

7. **Print head (when `print_head`).** Stdout, not logger — this is user-facing tabular output, not diagnostic noise:

   ```python
   if print_head:
       cols = [c for c in ('probe_id', 'gene_symbol', 'log2FC', 'adj_p_value') if c in de.columns]
       print(de[cols].head(_HEAD_N).to_string(index=False))
   ```

   Project four columns, not the full eight — the t-statistic and per-group means are useful in the CSV but clutter the terminal. Omit columns that don't exist (e.g. no `gene_symbol` when `annotation is None`).

8. **Return the frame.**

   ```python
   return de
   ```

### Edge cases to be aware of

- **All samples filtered out by `subset_by_group`.** Happens when the user typed the wrong `--group-a` / `--group-b` (e.g. `Tumor` vs the metadata's `tumor`). `differential_expression` catches this via its "fewer than 2 samples" check, but the message names the case-sensitive label the user passed. Acceptable — error is actionable.
- **Annotation has duplicate `gene_symbol` entries.** Different probes mapping to the same gene is normal for microarrays. The collapse step (step 5) handles this. If `collapse_to_gene=False`, the CSV will contain multiple rows for the same gene — also fine, that's what the caller asked for.
- **`expression` and `samples` index mismatch.** Already handled by `shared_utils.differential_expression`'s precondition check ("Expression columns must be a superset of the filtered sample IDs"). Don't re-validate here.
- **Re-running diffex with different group definitions** overwrites `de.csv`. Acceptable — `de.csv` is regenerable, not source of truth. Document this in [readme.md](readme.md).
- **Empty `de` after collapse.** Happens only if every probe has a NaN `gene_symbol` — implies broken annotation. The empty CSV is written and the print head is empty; subsequent volcano/heatmap calls will fail with "no rows" errors that are obvious enough not to need pre-emptive guarding.

### `ge.py` integration

The handler in `ge.py` (`_cmd_diffex` in the CLI section) currently calls `shared_utils.differential_expression` directly. Replace its body with one call into `run_diffex`:

```python
def _cmd_diffex(args: argparse.Namespace) -> None:
    expression, samples, annotation = _load_and_group(
        args.accession, args.group_col, args.group_a, args.group_b
    )
    diffex.run_diffex(
        expression, samples, annotation,
        args.group_a, args.group_b,
        output_dir=_accession_dir(args.accession),
    )
```

The `all` handler likewise drops its inline `detect_log_scale`/`normalize`/`differential_expression` block and calls `run_diffex(..., print_head=False)` — the head printout is noise inside the multi-step pipeline.

### Verification

Extend [test.py](test.py) with:

```python
import logging, os
from src import diffex

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
out = os.path.join('result', 'GSE19804')
de = diffex.run_diffex(exp, samp, ann, 'tumor', 'normal', output_dir=out)
print('shape:', de.shape, 'cols:', list(de.columns))
print('csv exists:', os.path.exists(os.path.join(out, 'de.csv')))
```

Expected for GSE19804 (lung cancer tumor vs normal, ~54k probes, 60+60 samples):

1. Log line `expression detected as log-scale; skipping log2 transform` (GSE19804 ships log2-scaled).
2. Log line `diffex: 54675 genes x 120 samples (tumor=60, normal=60)`.
3. Returned `de.shape` is `(54675, 8)` — 8 columns: `probe_id, gene_symbol, log2FC, mean_a, mean_b, t_stat, p_value, adj_p_value`.
4. `result/GSE19804/de.csv` exists, ~54k+1 lines (header + rows), file size in the low MBs.
5. Top-10 head printed to stdout includes well-known lung-cancer markers — `SPP1`, `MMP1`, `MMP12`, `WIF1`, `AGER` are all known to be in the top hits for this series.
6. Re-run with `collapse_to_gene=True`: row count drops from ~54675 to ~21000 (one row per unique gene symbol, minus probes with no symbol). Log line confirms the collapse.
7. Re-run with `print_head=False, output_dir=None`: nothing printed, no CSV written, just the in-memory frame returned. Confirms the side-effect-free path works.

## `src/volcano.py` — volcano plot

Inputs: a `de_results` DataFrame (or path to the `de.csv` written by `diffex.py`). One PNG output. Sibling module to [src/eda.py](src/eda.py) and [src/diffex.py](src/diffex.py); follows the same callable-entry, logger, no-try/except conventions.

### Design decisions

- **DataFrame in, path out.** `plot_volcano(de_results, output_path, ...)` — the caller is responsible for loading `de.csv` and computing the path. Mirrors [src/heatmap.py](src/heatmap.py) and keeps `volcano.py` ignorant of the `result/<accession>/...` scheme.
- **Three scatter calls, not one.** Plotting up/down/non-sig in three separate `ax.scatter` calls (instead of one call with a colors array) is what makes `ax.legend()` work without manual `Patch` wrangling — each call gets a `label=`, legend reads them off automatically.
- **Annotate with `ax.annotate`, not `ax.text`.** `annotate` lets you set `xytext=` in offset points so labels don't overlap their dots regardless of axis scaling. `text` would need manual unit math.
- **Clip on `adj_p_value`, not on `-log10`.** Tiny p-values (< 1e-300, possible from `ttest_ind` on ~50 samples) become `inf` after `-log10` and break `matplotlib`'s autoscale. Clip the input to `1e-300` before the log so the y-axis stays finite.
- **`plt.close(fig)` after save.** Matches `eda.py`. Without it, repeated CLI runs leak figures.

### Module header

```python
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
```

No `_ROOT`/sys.path block needed — this module imports nothing from `shared_utils`. Plain seaborn isn't imported either; volcano is pure matplotlib, no statistical aesthetics needed.

### Public entry function

```python
def plot_volcano(
    de_results: pd.DataFrame,
    output_path: str,
    adj_p_max: float = 0.05,
    abs_log2fc_min: float = 1.0,
    annotate_top: int = 10,
) -> None:
    """Render volcano plot to output_path. Defaults: p<0.05, |log2FC|>=1, top-10 labels."""
```

Three thresholds are caller-tunable because they're judgment calls — `0.05` and `1.0` are conventional but a sparse dataset may want `0.1` and `0.585` (1.5× fold change), a noisy one `0.01` and `2.0`. Don't bake them in.

### Step-by-step implementation

1. **Validate input columns.** Catch the missing-column case here with a clear message — `KeyError: 'log2FC'` from a downstream pandas op is opaque about which step asked for it:

   ```python
   required = {'log2FC', 'adj_p_value'}
   missing = required - set(de_results.columns)
   if missing:
       raise ValueError(
           f"de_results missing columns: {sorted(missing)}. "
           f"Expected output of shared_utils.differential_expression."
       )
   ```

   Don't require `gene_symbol` — fall back to `probe_id` for labels, fall back to row index if neither exists.

2. **Compute axes.** Drop rows with NaN p-values (genes that failed the t-test in `differential_expression`) — they have no defined position on the y-axis:

   ```python
   df = de_results.dropna(subset=['adj_p_value', 'log2FC']).copy()
   x = df['log2FC'].to_numpy()
   p_clipped = df['adj_p_value'].clip(lower=_P_FLOOR)
   y = -np.log10(p_clipped.to_numpy())
   ```

   `.copy()` because we'll add a transient `_label` column for annotation; don't mutate the caller's frame.

3. **Classify each row** into up / down / non-significant:

   ```python
   sig = df['adj_p_value'] < adj_p_max
   is_up = sig & (df['log2FC'] >= abs_log2fc_min)
   is_down = sig & (df['log2FC'] <= -abs_log2fc_min)
   is_nonsig = ~(is_up | is_down)
   ```

   Using `>=` / `<=` (not strict `>` / `<`) so a gene exactly at the threshold counts as up/down, not non-sig. `is_nonsig` covers everything else, including significant-but-small-effect genes — they're in the "p passed but effect size too small" zone, conventionally drawn grey.

4. **Build the figure.** Single axes, fixed aspect ratio (volcano plots are typically wider than tall):

   ```python
   fig, ax = plt.subplots(figsize=(8, 6))
   ax.scatter(x[is_nonsig], y[is_nonsig], s=_POINT_SIZE, alpha=_POINT_ALPHA,
              c=_COLOR_NONSIG, label='non-significant')
   ax.scatter(x[is_up], y[is_up], s=_POINT_SIZE, alpha=_POINT_ALPHA,
              c=_COLOR_UP, label=f'up (n={int(is_up.sum())})')
   ax.scatter(x[is_down], y[is_down], s=_POINT_SIZE, alpha=_POINT_ALPHA,
              c=_COLOR_DOWN, label=f'down (n={int(is_down.sum())})')
   ```

   Plot non-sig **first** so up/down dots end up on top — z-order matters, and the interesting dots being underneath the grey wash is the most common volcano-plot bug.

5. **Threshold reference lines** — three dashed greys:

   ```python
   ax.axhline(-np.log10(adj_p_max), color='grey', linestyle='--', linewidth=0.8)
   ax.axvline(abs_log2fc_min, color='grey', linestyle='--', linewidth=0.8)
   ax.axvline(-abs_log2fc_min, color='grey', linestyle='--', linewidth=0.8)
   ```

   Three calls (not a loop) — total six lines and reads top-to-bottom.

6. **Top-N annotation.** Pick the `annotate_top` rows with the smallest `adj_p_value` *among significant* genes and annotate each. Skip the step entirely if `annotate_top <= 0`:

   ```python
   if annotate_top > 0:
       label_col = 'gene_symbol' if 'gene_symbol' in df.columns else 'probe_id'
       sig_df = df[is_up | is_down].copy()
       sig_df['_y'] = -np.log10(sig_df['adj_p_value'].clip(lower=_P_FLOOR))
       top = sig_df.nsmallest(annotate_top, 'adj_p_value')
       for _, row in top.iterrows():
           label = row.get(label_col)
           if pd.isna(label) or label == '':
               label = row.get('probe_id', '')
           ax.annotate(
               str(label),
               xy=(row['log2FC'], row['_y']),
               xytext=(4, 4), textcoords='offset points',
               fontsize=7,
           )
   ```

   `nsmallest` is the right tool — `sort_values().head(N)` works but is `O(n log n)` for what should be an `O(n)` selection. The empty-label fallback (`pd.isna(label) or label == ''`) handles annotation rows where `gene_symbol` exists as a column but is missing for that probe.

7. **Axis labels, title, legend, save:**

   ```python
   ax.set_xlabel('log2 fold change (group_a − group_b)')
   ax.set_ylabel('−log10(adj p-value)')
   ax.set_title(
       f'Volcano: {int(is_up.sum())} up, {int(is_down.sum())} down '
       f'(p<{adj_p_max}, |log2FC|>={abs_log2fc_min})'
   )
   ax.legend(loc='upper right', fontsize=8)
   os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
   fig.savefig(output_path, **_SAVEFIG_KW)
   plt.close(fig)
   logger.info("wrote %s (up=%d, down=%d, n=%d)",
               output_path, int(is_up.sum()), int(is_down.sum()), len(df))
   ```

   `os.path.dirname(output_path) or '.'` covers both `/abs/path/volcano.png` and bare `volcano.png` — `dirname` of the latter is `''`, which `os.makedirs` rejects.

### Edge cases to be aware of

- **All NaN p-values** → `df` is empty after step 2's `dropna`. Three empty scatters and an empty plot result. Acceptable: the file still exists and the next step in the pipeline doesn't crash. Log a warning if `len(df) == 0` so the user notices.
- **No significant genes at the chosen thresholds** → `is_up.sum() == 0`, `is_down.sum() == 0`. Plot is all grey, title says `0 up, 0 down`. The user will likely re-run with looser thresholds — that's the right loop, don't auto-relax.
- **`adj_p_value` of exactly 0.0** → some statsmodels versions emit `0.0` for tiny p-values. `clip(lower=_P_FLOOR)` covers this.
- **Duplicate `gene_symbol`** in annotation candidates → multiple top probes for the same gene get labeled separately. Acceptable for v1; collapse-to-gene happens in `diffex.run_diffex(collapse_to_gene=True)` upstream if the user wants one label per gene.
- **`log2FC` is `±inf`** (zero variance in one group, mean of one group is zero) → matplotlib's autoscale silently expands to fit. Filter these out earlier in `differential_expression` if it becomes a real problem; not worth defending here.

### `ge.py` integration

The handler `_cmd_volcano` in [ge.py](ge.py) loads `de.csv` and calls `plot_volcano`:

```python
def _cmd_volcano(args: argparse.Namespace) -> None:
    de = pd.read_csv(_de_csv_path(args.accession, args.from_results))
    out_png = os.path.join(_accession_dir(args.accession), 'volcano.png')
    volcano.plot_volcano(
        de, out_png,
        adj_p_max=args.adj_p_max,
        abs_log2fc_min=args.abs_log2fc_min,
        annotate_top=args.annotate_top,
    )
```

The `all` handler passes the in-memory `de` frame returned by `run_diffex` — no CSV round-trip.

The argparse block for the `volcano` subcommand needs to be added to `_build_parser()` (mirror the `diffex` block already there):

```python
volc_p = sub.add_parser('volcano', help='render volcano plot from a de.csv')
volc_p.add_argument('--accession', required=True)
volc_p.add_argument('--from-results', help='path to de.csv (default: result/<acc>/de.csv)')
volc_p.add_argument('--adj-p-max', type=float, default=0.05)
volc_p.add_argument('--abs-log2fc-min', type=float, default=1.0)
volc_p.add_argument('--annotate-top', type=int, default=10)
```

### Verification

Extend [test.py](test.py) with:

```python
import logging, os
import pandas as pd
from src import volcano

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
de = pd.read_csv(os.path.join('result', 'GSE19804', 'de.csv'))
out_png = os.path.join('result', 'GSE19804', 'volcano.png')
volcano.plot_volcano(de, out_png)
print('volcano exists:', os.path.exists(out_png))
```

Expected for GSE19804 (lung cancer tumor vs normal, ~54k probes):

1. Log line `wrote result/GSE19804/volcano.png (up=~3000, down=~2000, n=54675)` — exact counts depend on filtering but tumor vs normal lung shows thousands of significant genes either side.
2. Open `volcano.png` — characteristic volcano shape: dense grey cloud at the bottom, two clear "wings" of red (upper right) and blue (upper left), tapering upward to a few hundred high-confidence genes.
3. Annotated labels include `SPP1`, `MMP1`, `MMP12`, `WIF1`, `AGER` — known lung-cancer markers near the top of the y-axis.
4. Three dashed reference lines visible: one horizontal at y=−log10(0.05)≈1.3, two vertical at x=±1.
5. Re-run with `annotate_top=0`: same plot, no labels — confirms the annotation block is opt-out cleanly.
6. Re-run with `adj_p_max=0.01, abs_log2fc_min=2.0`: fewer red/blue dots, title reflects new thresholds, threshold lines move accordingly.

## `src/heatmap.py` — clustered heatmap

Inputs: `expression` (already log-scaled, **not** z-scored — z-scoring happens here per row), `samples` with a `group` column, `de_results` (output of `differential_expression`), output path. One PNG output. Sibling module to [src/eda.py](src/eda.py), [src/diffex.py](src/diffex.py), and [src/volcano.py](src/volcano.py); follows the same callable-entry, logger, no-try/except conventions.

### Design decisions

- **DataFrame in, path out.** `plot_heatmap(expression, samples, de_results, output_path, ...)` — caller loads inputs and computes the path. Mirrors [src/volcano.py](src/volcano.py); keeps `heatmap.py` ignorant of the `result/<accession>/...` scheme and lets the `all` orchestrator pass in-memory frames without a CSV/parquet round-trip.
- **`sns.clustermap`, not `sns.heatmap`.** `clustermap` runs hierarchical clustering on rows and columns and draws both dendrograms in one call. Doing it manually with `scipy.cluster.hierarchy` plus `sns.heatmap` is ~30 lines of axis-juggling for the same picture.
- **Z-score rows, not columns.** Each row (gene) is centered and scaled independently so the colormap shows *relative* expression across samples. Without row z-scoring, a single high-magnitude probe washes out everything else into a single shade. Done via `shared_utils.normalize(mat, 'zscore')` so the rule (mean/std along axis=1, std==0 rows go NaN and get dropped) lives in one place — see [shared_utils.py](shared_utils.py)'s `normalize` notes.
- **Diverging colormap centered at 0.** `cmap='RdBu_r'`, `center=0` — after z-score, 0 is the row mean and the eye should read deviations symmetrically.
- **Column color bar over column dendrogram.** `col_colors=` paints a thin strip above the heatmap mapping each sample to its group. Lets the reader visually verify "did samples cluster by group?" without staring at GSM IDs. Two-class `sns.color_palette('Set2', 2)` keeps tumor/normal visually distinct from the heatmap's red/blue scale.
- **Drop unassigned samples explicitly here.** Not via `subset_by_group` — `samples` may already be group-labeled by the caller, but this module shouldn't crash if a NaN slipped through. `samples.dropna(subset=['group'])` is one line and self-documenting.
- **Height scales with `top`.** A 50-gene heatmap fits in 9–10 inches; a 200-gene one needs more. `figsize=(10, max(6, top*0.18))` keeps row labels legible without an enormous default. The `max(6, ...)` floor stops tiny `top` values producing a squashed strip.
- **`plt.close(g.fig)` after save.** Same leak-prevention rule as `eda.py` and `volcano.py`. `clustermap` returns a `ClusterGrid`, not a `Figure` — close `g.fig` explicitly.

### Module header

```python
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
```

The `_ROOT`/`sys.path` block matches `eda.py` — `heatmap.py` is the only plot module that pulls from `shared_utils` (`top_de_genes` for gene selection, `normalize` for row z-scoring). `volcano.py` doesn't because it operates on the already-computed `de_results` frame.

### Public entry function

```python
def plot_heatmap(
    expression: pd.DataFrame,
    samples: pd.DataFrame,
    de_results: pd.DataFrame,
    output_path: str,
    top: int = 50,
    adj_p_max: float = 0.05,
    abs_log2fc_min: float = 1.0,
) -> None:
    """Render clustered heatmap of top DE genes to output_path.
    Defaults: top 50 genes meeting p<0.05 and |log2FC|>=1."""
```

`adj_p_max` and `abs_log2fc_min` are forwarded to `top_de_genes` so the caller can loosen them if `top_de_genes` returns fewer rows than requested. Same defaults as `volcano.py` so both plots agree on what counts as "significant" by default.

### Step-by-step implementation

1. **Validate input columns.** Same defensive check as `volcano.py` — fail fast with a clear message rather than letting a downstream `KeyError` surface:

   ```python
   required_de = {'log2FC', 'adj_p_value', 'probe_id'}
   missing = required_de - set(de_results.columns)
   if missing:
       raise ValueError(
           f"de_results missing columns: {sorted(missing)}. "
           f"Expected output of shared_utils.differential_expression."
       )
   if 'group' not in samples.columns:
       raise ValueError(
           "samples missing 'group' column. "
           "Call shared_utils.assign_groups before plot_heatmap."
       )
   ```

   `gene_symbol` is *not* required — fall back to `probe_id` for row labels.

2. **Pick the gene set:**

   ```python
   top_genes = top_de_genes(
       de_results, n=top,
       adj_p_max=adj_p_max, abs_log2fc_min=abs_log2fc_min,
   )
   if top_genes.empty:
       logger.warning(
           "no genes pass thresholds (p<%s, |log2FC|>=%s); skipping heatmap",
           adj_p_max, abs_log2fc_min,
       )
       return
   ```

   Empty-set early return rather than writing a blank PNG — lets the `all` pipeline continue without crashing the next step, and the warning tells the user which knob to loosen. Don't auto-relax the thresholds.

3. **Subset rows (genes) and columns (samples).** Drop unassigned samples and re-align columns:

   ```python
   labeled = samples.dropna(subset=['group'])
   if labeled.empty:
       raise ValueError("no samples have an assigned group")
   probe_ids = top_genes['probe_id'].tolist()
   mat = expression.loc[probe_ids, labeled.index]
   ```

   `expression.loc[probe_ids, labeled.index]` does row + column subset in one indexing call. If a `probe_id` from `top_genes` is missing in `expression` (shouldn't happen — `de_results` was built from the same matrix — but possible if the caller passes a sliced `expression`), `.loc` raises `KeyError` immediately, which is the right failure.

4. **Drop zero-variance rows before z-scoring.** A row that's constant across the kept samples gives `std == 0` → division by zero → NaN row → `clustermap` crashes during the linkage step. Filter first:

   ```python
   row_std = mat.std(axis=1)
   keep = row_std > 0
   if not keep.all():
       dropped = (~keep).sum()
       logger.warning("dropping %d zero-variance rows before z-score", dropped)
       mat = mat.loc[keep]
   ```

   This is rare in practice — top DE genes by definition have variation — but the failure mode is a `LinAlgError` deep in scipy that's painful to debug.

5. **Z-score per row** via the shared helper:

   ```python
   mat_z = normalize(mat, 'zscore')
   ```

   `shared_utils.normalize` z-scores along `axis=1` (per row) and returns a copy with the same index/columns. Don't re-implement here — keeps the row-vs-column convention in one place.

6. **Build the column color bar.** Map each sample's `group` value to a color, returned as a `Series` indexed by sample so `clustermap` aligns it with columns:

   ```python
   groups = labeled.loc[mat_z.columns, 'group']
   group_levels = sorted(groups.unique())
   palette = sns.color_palette(_GROUP_PALETTE, len(group_levels))
   group_to_color = dict(zip(group_levels, palette))
   col_colors = groups.map(group_to_color)
   col_colors.name = 'group'
   ```

   `sorted(groups.unique())` makes the color assignment deterministic — same group → same color across runs, which matters when comparing PNGs by eye between threshold experiments. `col_colors.name = 'group'` labels the strip in the rendered figure.

7. **Relabel rows to gene symbol where available.** Probe IDs (`'1554567_a_at'`) are unreadable; gene symbols (`'SPP1'`) are what the reader actually wants:

   ```python
   if 'gene_symbol' in top_genes.columns:
       label_map = top_genes.set_index('probe_id')['gene_symbol']
       new_index = [
           label_map.get(p) if isinstance(label_map.get(p), str)
                            and label_map.get(p) != ''
           else p
           for p in mat_z.index
       ]
       mat_z = mat_z.copy()
       mat_z.index = new_index
   ```

   Per-row fallback (not "all-or-nothing") because annotation tables routinely have `gene_symbol` for most probes and missing values for a few — a probe with no symbol still deserves a label (its probe ID).

8. **Clustermap call:**

   ```python
   height = max(6.0, min(20.0, top * 0.18))
   g = sns.clustermap(
       mat_z,
       cmap=_CMAP,
       center=0,
       col_colors=col_colors,
       figsize=(10, height),
       xticklabels=True,
       yticklabels=True,
       dendrogram_ratio=(0.12, 0.18),
       cbar_pos=(0.02, 0.85, 0.03, 0.1),
       linewidths=0,
   )
   ```

   - `min(20.0, ...)` caps the height so a `top=200` run doesn't produce a 36-inch figure that matplotlib rejects.
   - `dendrogram_ratio=(0.12, 0.18)` shrinks both dendrograms slightly — the default eats too much canvas.
   - `cbar_pos` puts the colorbar in the top-left, out of the way; the default position overlaps the column dendrogram on tall figures.
   - `linewidths=0` because cell borders are visual noise at 50+ rows.

9. **Tweak tick labels and title:**

   ```python
   g.ax_heatmap.set_xticklabels(
       g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=6,
   )
   g.ax_heatmap.set_yticklabels(
       g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=7,
   )
   g.ax_heatmap.set_xlabel('sample')
   g.ax_heatmap.set_ylabel('gene')
   g.fig.suptitle(
       f'Top {len(mat_z)} DE genes (z-scored rows) — '
       f'{len(group_levels)} groups',
       y=1.02, fontsize=10,
   )
   ```

   Rotated, tiny xticks because GSM IDs are long and there are many of them; small but readable yticks for gene symbols. `y=1.02` lifts the title above the column dendrogram.

10. **Save and close:**

    ```python
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    g.savefig(output_path, **_SAVEFIG_KW)
    plt.close(g.fig)
    logger.info(
        "wrote %s (genes=%d, samples=%d, groups=%d)",
        output_path, mat_z.shape[0], mat_z.shape[1], len(group_levels),
    )
    ```

    `os.path.dirname(output_path) or '.'` covers the bare-filename case, same as `volcano.py`. `plt.close(g.fig)` — `ClusterGrid` doesn't have its own close method.

### Edge cases to be aware of

- **Empty `top_genes`** → handled in step 2, log + return. The PNG is *not* created; the calling pipeline tolerates this.
- **Single group in `samples`** → `len(group_levels) == 1`, color bar is a single uniform stripe (still informative as a "labeled" indicator). Clustermap still runs — sample dendrogram just clusters by expression similarity within the one group.
- **All probes have NaN `gene_symbol`** → fallback in step 7 keeps probe IDs as labels; figure renders fine, just less readable.
- **`top` larger than the number of significant genes** → `top_de_genes` returns whatever clears the thresholds; `mat_z` has fewer rows than requested; figure height shrinks to the `max(6, ...)` floor.
- **NaNs in `expression`** for some probe × sample cells → propagates through z-score; `clustermap` will refuse to compute linkage on NaN rows. If this surfaces in real data, fill within `differential_expression` upstream rather than papering over it here.
- **Duplicate `gene_symbol`** across the top set (e.g. two probes for `MMP1`) → both rows keep their symbol label; the heatmap shows two `MMP1` rows. Fine for v1; collapse-to-gene happens in `diffex.run_diffex(collapse_to_gene=True)` if the user wants one row per gene.
- **`expression` indexed by gene symbol, not probe ID** (caller passed a collapsed frame plus an uncollapsed `de_results`) → `expression.loc[probe_ids, ...]` raises `KeyError` in step 3. The error message names the missing probes, which is enough to diagnose.

### `ge.py` integration

The handler `_cmd_heatmap` in [ge.py](ge.py) loads everything `plot_heatmap` needs and calls it:

```python
def _cmd_heatmap(args: argparse.Namespace) -> None:
    expression, samples, annotation = shared_utils.load_geo_dataset(
        args.accession, cache_dir=_data_dir(),
    )
    samples = shared_utils.assign_groups(
        samples,
        source_col=args.group_col,
        substrings={args.group_a: args.group_a, args.group_b: args.group_b},
    )
    de = pd.read_csv(_de_csv_path(args.accession, args.from_results))
    out_png = os.path.join(_accession_dir(args.accession), 'heatmap.png')
    heatmap.plot_heatmap(
        expression, samples, de, out_png,
        top=args.top,
        adj_p_max=args.adj_p_max,
        abs_log2fc_min=args.abs_log2fc_min,
    )
```

The `all` handler passes the in-memory `expression`, `samples` (already group-assigned), and `de` returned by `run_diffex` — no GEO reload, no CSV round-trip.

The argparse block for the `heatmap` subcommand mirrors `volcano`'s:

```python
hm_p = sub.add_parser('heatmap', help='render clustered heatmap of top DE genes')
hm_p.add_argument('--accession', required=True)
hm_p.add_argument('--group-col', required=True)
hm_p.add_argument('--group-a', required=True)
hm_p.add_argument('--group-b', required=True)
hm_p.add_argument('--from-results', help='path to de.csv (default: result/<acc>/de.csv)')
hm_p.add_argument('--top', type=int, default=50)
hm_p.add_argument('--adj-p-max', type=float, default=0.05)
hm_p.add_argument('--abs-log2fc-min', type=float, default=1.0)
```

`--group-col`/`--group-a`/`--group-b` are required here (unlike `volcano`, which only needs the precomputed `de.csv`) because `heatmap` needs the sample-to-group mapping to draw `col_colors`.

### Verification

Extend [test.py](test.py) with:

```python
import logging, os
import pandas as pd
from src import heatmap
from shared_utils import load_geo_dataset, assign_groups

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
expression, samples, annotation = load_geo_dataset('GSE19804', cache_dir='data')
samples = assign_groups(
    samples,
    source_col='source_name_ch1',
    substrings={'tumor': 'tumor', 'normal': 'normal'},
)
de = pd.read_csv(os.path.join('result', 'GSE19804', 'de.csv'))
out_png = os.path.join('result', 'GSE19804', 'heatmap.png')
heatmap.plot_heatmap(expression, samples, de, out_png, top=50)
print('heatmap exists:', os.path.exists(out_png))
```

Expected for GSE19804 (lung cancer tumor vs normal, ~120 samples):

1. Log line `wrote result/GSE19804/heatmap.png (genes=50, samples=~118, groups=2)` — sample count slightly under 120 if a couple of GSMs failed substring matching.
2. Open `heatmap.png` — column dendrogram splits cleanly into two large clades; the column color strip above the heatmap shows that split aligns with tumor/normal (i.e. the dendrogram is *not* interleaving groups).
3. Row dendrogram shows two main blocks — up-regulated genes (red across tumor samples, blue across normal) and down-regulated genes (the inverse pattern).
4. Visible row labels include `SPP1`, `MMP1`, `MMP12`, `WIF1`, `AGER` — same lung-cancer markers that `volcano.png` annotates, sanity-checking that both plots are reading the same `de_results`.
5. Re-run with `top=200`: figure is taller, individual row labels become harder to read but the two-block structure persists. Confirms the `top` parameter scales correctly.
6. Re-run with `adj_p_max=1.0, abs_log2fc_min=0.0`: thresholds are effectively disabled, `top_de_genes` ranks purely by `|log2FC|`, heatmap still renders and the strongest genes still drive the clustering.
7. Re-run after artificially blanking the `group` column on one sample: that sample is dropped (visible by sample count in log line), the rest of the figure is unchanged.

## `src/compare.py` — cross-dataset comparison

Inputs: two `de_results` DataFrames (already collapsed to one row per `gene_symbol`), labels for each dataset (e.g. accession strings), output directory. Three outputs: `shared_de_genes.csv`, `log2fc_scatter.png`, `overlap_summary.txt`. Sibling module to [src/diffex.py](src/diffex.py), [src/volcano.py](src/volcano.py), and [src/heatmap.py](src/heatmap.py); follows the same callable-entry, logger, no-try/except conventions.

### Design decisions

- **DataFrames in, directory out.** `run_compare(de_a, de_b, output_dir, label_a, label_b, ...)` — caller is responsible for running `diffex.run_diffex(..., collapse_to_gene=True)` on each dataset and passing the frames in. Mirrors the rest of `src/`: the module is ignorant of GEO loading and the `result/...` directory scheme. Lets the CLI handler (`_cmd_compare`) own I/O and lets `compare.py` be pure analysis.
- **Collapse-to-gene happens upstream, not here.** `run_diffex(collapse_to_gene=True)` already implements the "smallest p wins per gene_symbol" rule (see step 5 of `diffex.py`). Re-implementing it here would duplicate logic and risk drift. `compare.py` validates that its inputs are already collapsed (one row per `gene_symbol`, no NaN symbols) and raises if not — fail loud rather than silently produce a half-correct join.
- **Inner join on `gene_symbol` for the scatter; outer for the set math.** The scatter plot needs both `log2FC` values, so it lives on the inner join. The Jaccard / "in A only" / "in B only" counts need the union, so the significance-set comparison uses each dataset's full collapsed list.
- **Three output files, not one.** A CSV for downstream re-use, a PNG for visual inspection, a TXT for at-a-glance numbers. Mirrors the EDA module's "one summary text + plots" split — no single output is responsible for everything.
- **Pearson `r` and Spearman `ρ` both in the scatter title.** Pearson reports linear concordance; Spearman reports rank concordance and is robust to a few extreme-fold-change outliers that one platform measures and the other doesn't. Quoting both lets the reader judge whether an apparently-low Pearson is driven by genuine disagreement or by a few outlier genes.
- **Annotate by "combined rank", not raw `|log2FC|`.** A gene with huge `log2FC` in dataset A but `log2FC ≈ 0` in dataset B isn't interesting for a *comparison* — interesting genes are large in both. Combined rank = `rank(|log2FC_a|) + rank(|log2FC_b|)`, smallest sum wins.
- **No try/except.** Same convention as the rest of `src/`. Let `pd.merge`'s key errors and `scipy.stats`'s NaN-input errors propagate.

### Module header

```python
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
```

No `_ROOT`/`sys.path` block — `compare.py` doesn't import from `shared_utils`. All inputs are pre-computed frames.

### Public entry function

```python
def run_compare(
    de_a: pd.DataFrame,
    de_b: pd.DataFrame,
    output_dir: str,
    label_a: str,
    label_b: str,
    *,
    adj_p_max: float = _DEFAULT_ADJ_P_MAX,
    abs_log2fc_min: float = _DEFAULT_ABS_LOG2FC_MIN,
    annotate_top: int = _ANNOTATE_TOP,
) -> pd.DataFrame:
    """Compare two collapsed-to-gene de_results frames.

    Writes three files to output_dir and returns the joined frame
    (inner join on gene_symbol).
    """
```

`label_a` / `label_b` (typically GEO accessions like `'GSE19804'`) become the column suffixes in the joined frame and the axis labels on the scatter. Returning the joined frame lets the `all` orchestrator (or a notebook caller) inspect the result without re-reading the CSV.

### Step-by-step implementation

1. **Validate input columns and collapse status.** Both frames must have the four columns `compare.py` actually uses, and must be collapsed to one row per gene_symbol — mismatched assumptions here produce bad joins:

   ```python
   required = {'gene_symbol', 'log2FC', 'adj_p_value'}
   for name, df in (('de_a', de_a), ('de_b', de_b)):
       missing = required - set(df.columns)
       if missing:
           raise ValueError(
               f"{name} missing columns: {sorted(missing)}. "
               f"Expected output of run_diffex(..., collapse_to_gene=True)."
           )
       if df['gene_symbol'].isna().any():
           raise ValueError(
               f"{name} contains NaN gene_symbol entries. "
               f"Pass collapse_to_gene=True to run_diffex."
           )
       if df['gene_symbol'].duplicated().any():
           dup_n = int(df['gene_symbol'].duplicated().sum())
           raise ValueError(
               f"{name} has {dup_n} duplicate gene_symbol rows. "
               f"Pass collapse_to_gene=True to run_diffex."
           )
   ```

   Three checks, three distinct error messages — saves a round of "wait, which assumption did I break?" debugging.

2. **Inner join on `gene_symbol`** with explicit suffixes. Keep only the columns we actually use:

   ```python
   cols = ['gene_symbol', 'log2FC', 'adj_p_value']
   joined = de_a[cols].merge(
       de_b[cols], on='gene_symbol', how='inner',
       suffixes=(f'_{label_a}', f'_{label_b}'),
   )
   logger.info(
       "joined %d × %d genes -> %d shared on gene_symbol",
       len(de_a), len(de_b), len(joined),
   )
   ```

   Suffix with the label, not `_a`/`_b` — when the user opens the CSV in Excel, `log2FC_GSE19804` is self-documenting; `log2FC_a` requires checking the header against memory.

3. **Drop rows with NaN in any of the four numeric columns** before computing correlation / sign concordance — `scipy.stats.pearsonr` raises on NaN inputs:

   ```python
   numeric_cols = [
       f'log2FC_{label_a}', f'log2FC_{label_b}',
       f'adj_p_value_{label_a}', f'adj_p_value_{label_b}',
   ]
   joined = joined.dropna(subset=numeric_cols).reset_index(drop=True)
   if joined.empty:
       raise ValueError(
           "no genes survived inner join + NaN drop — "
           "check that both inputs are collapsed and overlap on gene_symbol"
       )
   ```

4. **Significance flags and the four-way classification.** Significant in both / A only / B only / neither — drives both the scatter coloring and the summary counts:

   ```python
   sig_a_full = (de_a['adj_p_value'] < adj_p_max) & (de_a['log2FC'].abs() >= abs_log2fc_min)
   sig_b_full = (de_b['adj_p_value'] < adj_p_max) & (de_b['log2FC'].abs() >= abs_log2fc_min)
   set_a = set(de_a.loc[sig_a_full, 'gene_symbol'])
   set_b = set(de_b.loc[sig_b_full, 'gene_symbol'])

   joined['sig_a'] = (
       (joined[f'adj_p_value_{label_a}'] < adj_p_max)
       & (joined[f'log2FC_{label_a}'].abs() >= abs_log2fc_min)
   )
   joined['sig_b'] = (
       (joined[f'adj_p_value_{label_b}'] < adj_p_max)
       & (joined[f'log2FC_{label_b}'].abs() >= abs_log2fc_min)
   )
   ```

   `set_a` / `set_b` are computed from the **full** collapsed lists (not the inner-joined subset) so the Jaccard reflects genes-significant-in-A regardless of whether they appear in B's platform — that's the biologically honest count.

5. **Write `shared_de_genes.csv`** — genes significant in both. Sorted by combined rank so the most concordant hits land at the top:

   ```python
   shared = joined[joined['sig_a'] & joined['sig_b']].copy()
   shared['rank_a'] = shared[f'log2FC_{label_a}'].abs().rank(ascending=False)
   shared['rank_b'] = shared[f'log2FC_{label_b}'].abs().rank(ascending=False)
   shared['combined_rank'] = shared['rank_a'] + shared['rank_b']
   shared = (
       shared.sort_values('combined_rank')
             .drop(columns=['rank_a', 'rank_b'])
             .reset_index(drop=True)
   )
   os.makedirs(output_dir, exist_ok=True)
   shared.to_csv(
       os.path.join(output_dir, _SHARED_CSV_NAME),
       index=False, float_format='%.6g',
   )
   logger.info("wrote %s (%d genes)", _SHARED_CSV_NAME, len(shared))
   ```

   `'%.6g'` matches the format used in `diffex.py`'s CSV emission — keeps tiny adjusted p-values readable.

6. **Compute correlations on the joined set** (not just the both-significant subset — correlations are most meaningful over the full overlap):

   ```python
   x = joined[f'log2FC_{label_a}'].to_numpy()
   y = joined[f'log2FC_{label_b}'].to_numpy()
   pearson_r, _ = pearsonr(x, y)
   spearman_rho, _ = spearmanr(x, y)
   ```

   Discard the p-values from `pearsonr` / `spearmanr` — with tens of thousands of genes the null is rejected at meaningless significance levels, so the correlation magnitude is what matters.

7. **Build the scatter.** Three z-ordered scatter calls (same pattern as `volcano.py` step 4 — non-significant first so colored dots land on top):

   ```python
   fig, ax = plt.subplots(figsize=(7, 7))
   neither = ~(joined['sig_a'] | joined['sig_b'])
   one_only = (joined['sig_a'] ^ joined['sig_b'])
   both = joined['sig_a'] & joined['sig_b']

   ax.scatter(x[neither], y[neither], s=6, alpha=0.5,
              c=_COLOR_NEITHER, label=f'neither (n={int(neither.sum())})')
   ax.scatter(x[one_only], y[one_only], s=8, alpha=0.7,
              c=_COLOR_ONE, label=f'one only (n={int(one_only.sum())})')
   ax.scatter(x[both], y[both], s=10, alpha=0.85,
              c=_COLOR_BOTH, label=f'both (n={int(both.sum())})')
   ```

8. **Reference lines and a 1:1 diagonal.** The diagonal is the eye's anchor for "do the two platforms agree on direction *and* magnitude?":

   ```python
   lim = float(max(np.abs(x).max(), np.abs(y).max())) * 1.05
   ax.set_xlim(-lim, lim)
   ax.set_ylim(-lim, lim)
   ax.axhline(0, color='grey', linewidth=0.6, linestyle='--')
   ax.axvline(0, color='grey', linewidth=0.6, linestyle='--')
   ax.plot([-lim, lim], [-lim, lim], color='grey', linewidth=0.6, linestyle=':')
   ```

   Square axes (`figsize=(7, 7)` + symmetric limits) so a slope-1 line is actually 45°.

9. **Annotate the top concordant genes** — `annotate_top` rows with the smallest `combined_rank` among the both-significant set:

   ```python
   if annotate_top > 0 and not shared.empty:
       top = shared.head(annotate_top)
       for _, row in top.iterrows():
           ax.annotate(
               str(row['gene_symbol']),
               xy=(row[f'log2FC_{label_a}'], row[f'log2FC_{label_b}']),
               xytext=(4, 4), textcoords='offset points',
               fontsize=7,
           )
   ```

   `shared` is already sorted by `combined_rank` from step 5, so `.head(annotate_top)` is correct without re-sorting. Skip cleanly if `annotate_top <= 0` or `shared` is empty.

10. **Title, labels, legend, save:**

    ```python
    ax.set_xlabel(f'log2FC ({label_a})')
    ax.set_ylabel(f'log2FC ({label_b})')
    ax.set_title(
        f'log2FC concordance: Pearson r={pearson_r:.3f}, '
        f"Spearman ρ={spearman_rho:.3f} "
        f"(n={len(joined)})"
    )
    ax.legend(loc='upper left', fontsize=8)
    fig.savefig(os.path.join(output_dir, _SCATTER_PNG_NAME), **_SAVEFIG_KW)
    plt.close(fig)
    logger.info("wrote %s", _SCATTER_PNG_NAME)
    ```

    `ρ` (Greek rho) renders correctly in matplotlib without needing a TeX-mode title. Stick to ASCII in the source so the file stays grep-friendly.

11. **Build and write the summary text.** Plain text, key-value lines — easy to diff between runs and parse with `awk`/`cut` if the user wants to script over results:

    ```python
    only_a = set_a - set_b
    only_b = set_b - set_a
    both_set = set_a & set_b
    union = set_a | set_b
    jaccard = len(both_set) / len(union) if union else 0.0

    if both_set:
        signs_a = np.sign(de_a.set_index('gene_symbol').loc[list(both_set), 'log2FC'])
        signs_b = np.sign(de_b.set_index('gene_symbol').loc[list(both_set), 'log2FC'])
        same_sign = int((signs_a == signs_b).sum())
        concordance = same_sign / len(both_set)
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
        f"significant in both:    {len(both_set)}",
        f"only in {label_a}:       {len(only_a)}",
        f"only in {label_b}:       {len(only_b)}",
        "",
        f"jaccard:               {jaccard:.4f}",
        f"sign concordance:      {concordance:.4f} ({same_sign}/{len(both_set)})",
        f"pearson r (joined):    {pearson_r:.4f}",
        f"spearman rho (joined): {spearman_rho:.4f}",
    ]
    summary_path = os.path.join(output_dir, _SUMMARY_TXT_NAME)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    logger.info("wrote %s", _SUMMARY_TXT_NAME)
    ```

    Trailing newline because most editors expect it. `encoding='utf-8'` for Windows safety — the rest of the project assumes UTF-8.

12. **Return the joined frame:**

    ```python
    return joined
    ```

### Edge cases to be aware of

- **Empty intersection on `gene_symbol`** (different platforms with no shared symbols, or one input had its symbols stripped) → step 3 raises `ValueError`. The error message is actionable; better than three blank output files.
- **One dataset has zero significant genes** at the chosen thresholds → `set_a` (or `set_b`) is empty, `jaccard` = 0, `shared_de_genes.csv` is header-only. Scatter still renders with grey + one color of dots. The summary tells the user immediately what's happening.
- **All both-significant genes have the same direction** → `concordance = 1.0`. Boring but correct; happens with strongly-conserved signatures (e.g. tumor vs normal lung across two cohorts).
- **`adj_p_value` of exactly `0.0`** in either dataset → flagged significant by the `<` test, no special handling needed (unlike the volcano's `-log10` clip — `compare.py` doesn't log-transform p-values).
- **Duplicate `gene_symbol`** in inputs → step 1 raises. We don't silently dedupe; the user has the wrong shape and needs to fix `run_diffex`.
- **Massive log2FC outliers** (zero variance in one group → `±inf`) → caught by step 3's NaN drop only if they are NaN, not if they are `±inf`. If `inf` reaches the scatter, matplotlib autoscales to a useless figure. If this surfaces in real data, filter `±inf` in `differential_expression` upstream rather than papering over it here — same call as `volcano.py`.
- **Re-running compare** overwrites all three files. Acceptable — they're regenerable.

### `ge.py` integration

The handler `_cmd_compare` loads two datasets, runs `diffex` with collapse on each, and dispatches to `run_compare`:

```python
def _cmd_compare(args: argparse.Namespace) -> None:
    de_a = _diffex_for(args.accession_a, args)
    de_b = _diffex_for(args.accession_b, args)
    out_dir = os.path.join(
        _result_root(),
        f'compare_{args.accession_a}_vs_{args.accession_b}',
    )
    compare.run_compare(
        de_a, de_b, out_dir,
        label_a=args.accession_a, label_b=args.accession_b,
        adj_p_max=args.adj_p_max,
        abs_log2fc_min=args.abs_log2fc_min,
    )

def _diffex_for(accession: str, args: argparse.Namespace) -> pd.DataFrame:
    expression, samples, annotation = shared_utils.load_geo_dataset(
        accession, cache_dir=_data_dir(),
    )
    samples = shared_utils.assign_groups(
        samples,
        source_col=args.group_col,
        substrings={args.group_a: args.group_a, args.group_b: args.group_b},
    )
    return diffex.run_diffex(
        expression, samples, annotation,
        args.group_a, args.group_b,
        output_dir=None,             # don't overwrite per-dataset de.csv here
        collapse_to_gene=True,       # required by run_compare
        print_head=False,
    )
```

`output_dir=None` because the per-accession `de.csv` may already exist from a prior `ge.py diffex` run — don't overwrite it with the collapsed version. The collapsed frames live only in memory for the comparison.

The argparse block:

```python
cmp_p = sub.add_parser('compare', help='cross-dataset DE comparison')
cmp_p.add_argument('--accession-a', required=True)
cmp_p.add_argument('--accession-b', required=True)
cmp_p.add_argument('--group-col', required=True,
                   help='metadata column used for both datasets')
cmp_p.add_argument('--group-a', required=True)
cmp_p.add_argument('--group-b', required=True)
cmp_p.add_argument('--adj-p-max', type=float, default=0.05)
cmp_p.add_argument('--abs-log2fc-min', type=float, default=1.0)
```

`--group-col` is shared across both datasets, but the substrings used to identify each group can differ — GSE19804's `source_name_ch1` says `"... tumor ..."` while GSE10072's says `"Adenocarcinoma of the Lung"`. To handle that, the CLI accepts four optional override flags: `--group-a-substring-a`, `--group-a-substring-b`, `--group-b-substring-a`, `--group-b-substring-b`. Each defaults to its corresponding `--group-a` / `--group-b` value, so users with consistent wording across cohorts don't need to set anything. Both datasets still get assigned `group='tumor'` / `group='normal'` (or whatever labels are passed) so `run_compare`'s join works unchanged. Example for the GSE19804/GSE10072 pair: `--group-a tumor --group-b normal --group-a-substring-b adenocarcinoma`.

### Verification

Extend [test.py](test.py) with:

```python
import logging, os
from src import diffex, compare
from shared_utils import load_geo_dataset, assign_groups

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def collapsed_de(accession, tumor_substring='tumor'):
    expression, samples, annotation = load_geo_dataset(accession, cache_dir='data')
    samples = assign_groups(
        samples,
        source_col='source_name_ch1',
        substrings={'tumor': tumor_substring, 'normal': 'normal'},
    )
    return diffex.run_diffex(
        expression, samples, annotation, 'tumor', 'normal',
        output_dir=None, collapse_to_gene=True, print_head=False,
    )

# GSE19804 uses 'tumor' in source_name_ch1; GSE10072 uses 'Adenocarcinoma of the Lung'.
# Both keep the assigned group label as 'tumor' so the downstream join lines up.
de_a = collapsed_de('GSE19804')
de_b = collapsed_de('GSE10072', tumor_substring='adenocarcinoma')
out = os.path.join('result', 'compare_GSE19804_vs_GSE10072')
joined = compare.run_compare(de_a, de_b, out,
                             label_a='GSE19804', label_b='GSE10072')
print('joined shape:', joined.shape)
for fname in ('shared_de_genes.csv', 'log2fc_scatter.png', 'overlap_summary.txt'):
    print(fname, 'exists:', os.path.exists(os.path.join(out, fname)))
```

Expected for GSE19804 vs GSE10072 (both lung cancer tumor vs normal):

1. Log line `joined ~21000 × ~13000 genes -> ~12000 shared on gene_symbol` — exact counts depend on annotation completeness, but the inner-join size is bounded by the smaller of the two collapsed lists.
2. Three files appear in `result/compare_GSE19804_vs_GSE10072/`.
3. `overlap_summary.txt` shows `pearson r` and `spearman rho` both around `0.7–0.85` — strong concordance is expected because the two studies measure the same biology on related platforms.
4. `sign concordance` is around `0.95+` among both-significant genes — when both studies call a gene differentially expressed, they nearly always agree on direction.
5. `shared_de_genes.csv` first rows include known lung-cancer markers — `SPP1`, `MMP1`, `MMP12`, `WIF1`, `AGER` — the same set that volcano/heatmap surfaced per-dataset.
6. `log2fc_scatter.png` — points cluster along the y=x diagonal; red (both-significant) dots concentrate in the upper-right and lower-left quadrants; few red dots straddle the off-diagonal quadrants (which would indicate disagreement on direction).
7. Re-run with `adj_p_max=0.01, abs_log2fc_min=2.0`: scatter is unchanged in shape, but the red/orange/grey ratio shifts toward grey; `shared_de_genes.csv` shrinks; correlations on the joined set are unchanged (they're computed over the full overlap, not the significant subset).

## Output directory layout (runtime)

```
analyze_gene_expression/
  data/
    GSE19804_family.soft.gz        # GEOparse cache
    GSE10072_family.soft.gz
  result/
    GSE19804/
      eda/
        summary.txt
        sample_boxplots.png
        sample_correlation_heatmap.png
        pca_scatter.png
        gene_variance_hist.png
      de.csv
      volcano.png
      heatmap.png
    compare_GSE19804_vs_GSE10072/
      shared_de_genes.csv
      log2fc_scatter.png
      overlap_summary.txt
```

Add `data/` and `result/` to `.gitignore` (or the project's equivalent — repo currently has no root-level `.gitignore`; note in readme that these are build artifacts).

## Testing datasets

Recommend these public GEO series for manual verification — both small, fast to download, and have a clean two-group structure:

- **GSE19804** — 120 samples, lung cancer (tumor vs adjacent normal), Affymetrix HG-U133 Plus 2.0. Well-studied, strong DE signal.
- **GSE10072** — 107 samples, also lung cancer tumor/normal on the same platform family. Good partner for the `compare` subcommand — overlapping biology, different cohort.

These are suggestions for manual smoke testing, not hard-coded in the app.

## Critical files to create

- [ge.py](ge.py) — fill the empty stub (CLI entry).
- [shared_utils.py](shared_utils.py) — new.
- [requirements.txt](requirements.txt) — new.
- [src/eda.py](src/eda.py) — new.
- [src/diffex.py](src/diffex.py) — new.
- [src/volcano.py](src/volcano.py) — new.
- [src/heatmap.py](src/heatmap.py) — new.
- [src/compare.py](src/compare.py) — new.
- [readme.md](readme.md) — usage docs, mirroring [../Covid-Test/readme.md](../Covid-Test/readme.md).

Reference (do not modify):
- [../Covid-Test/shared_utils.py](../Covid-Test/shared_utils.py) — helper-module style.
- [../Covid-Test/src/](../Covid-Test/src/) — per-script module style.
- [../protein_visualization/requirements.txt](../protein_visualization/requirements.txt) — `GEOparse` precedent.

## Verification

After implementation, verify end-to-end against a real GEO series:

1. `pip install -r analyze_gene_expression/requirements.txt`.
2. `python analyze_gene_expression/ge.py eda --accession GSE19804`
   - Confirms GEOparse download works and writes 4 PNGs + `summary.txt` to `result/GSE19804/eda/`.
   - Open `pca_scatter.png` — tumor vs normal should separate visibly on PC1 or PC2.
3. `python analyze_gene_expression/ge.py diffex --accession GSE19804 --group-col source_name_ch1 --group-a tumor --group-b normal`
   - Writes `de.csv`. Spot-check: well-known lung cancer markers (e.g. `SPP1`, `MMP1`) should appear near the top.
4. `python analyze_gene_expression/ge.py volcano --accession GSE19804 --from-results result/GSE19804/de.csv`
   - Open `volcano.png` — should show the characteristic volcano shape with red/blue wings.
5. `python analyze_gene_expression/ge.py heatmap --accession GSE19804 --group-col source_name_ch1 --group-a tumor --group-b normal --from-results result/GSE19804/de.csv --top 50`
   - Open `heatmap.png` — dendrogram should cluster tumor samples away from normal samples.
6. `python analyze_gene_expression/ge.py compare --accession-a GSE19804 --accession-b GSE10072 --group-col source_name_ch1 --group-a tumor --group-b normal --group-a-substring-b adenocarcinoma`
   - `log2fc_scatter.png` should show positive Pearson r — the same biology across cohorts.
7. Re-run step 2 — should hit the GEOparse cache and finish in seconds (no re-download).

## Out of scope for v1

Called out so the scope is clear:
- RNA-seq count modeling (DESeq2 / edgeR / limma-voom) — v1 assumes microarray-style continuous expression already on log-scale or safely log2-transformable.
- Gene set enrichment (GSEA / ORA).
- Batch-effect correction (ComBat / limma::removeBatchEffect).
- Cross-platform probe-to-gene reconciliation beyond simple `gene_symbol` joining.
- Interactive dashboard — static PNGs only.
