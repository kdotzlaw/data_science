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
python ge.py diffex   --accession GSE19804 --group-col 'characteristics_ch1' --group-a 'tumor' --group-b 'normal'
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

1. Call `shared_utils.load_geo_dataset`.
2. Apply `normalize(expression, 'log2')` if the data isn't already log-scaled (reuse the EDA heuristic).
3. Call `shared_utils.assign_groups` + `shared_utils.differential_expression`.
4. Write `result/<accession>/de.csv` — full table, one row per gene.
5. Print top-10 by `adj_p_value` to stdout.

## `src/volcano.py` — volcano plot

Inputs: a `de_results` DataFrame (or path to the `de.csv` written by `diffex.py`).

```python
def plot_volcano(
    de_results: pd.DataFrame,
    output_path: str,
    adj_p_max: float = 0.05,
    abs_log2fc_min: float = 1.0,
    annotate_top: int = 10,
) -> None
```

Implementation:
- `x = de_results['log2FC']`, `y = -np.log10(de_results['adj_p_value'].clip(lower=1e-300))` (clip to avoid `-inf` for p≈0).
- Classify each gene:
  - grey: non-significant
  - red: `adj_p_value < adj_p_max` and `log2FC > abs_log2fc_min` (up)
  - blue: `adj_p_value < adj_p_max` and `log2FC < -abs_log2fc_min` (down)
- `plt.scatter` with `s=8, alpha=0.6` per category (three calls so legend works).
- Dashed threshold lines at `±abs_log2fc_min` (vertical) and `-log10(adj_p_max)` (horizontal).
- Annotate top-N by `adj_p_value` with `gene_symbol` (fall back to `probe_id`) using `matplotlib.text` + a small offset.
- `fig.savefig(output_path, dpi=200, bbox_inches='tight')`.

## `src/heatmap.py` — clustered heatmap

Inputs: `expression` (normalized), `samples`, `de_results`, output path.

```python
def plot_heatmap(
    expression: pd.DataFrame,
    samples: pd.DataFrame,
    de_results: pd.DataFrame,
    output_path: str,
    top: int = 50,
) -> None
```

Implementation:
- `top_genes = shared_utils.top_de_genes(de_results, n=top)` — pick the gene set.
- `mat = expression.loc[top_genes['probe_id']]` — subset rows.
- Keep only samples with an assigned group: `mat = mat[samples.dropna(subset=['group']).index]`.
- z-score rows via `shared_utils.normalize(mat, 'zscore')` — standard heatmap convention so colors show relative expression.
- Column color bar: map `samples['group']` to two colors via `sns.color_palette('Set2', 2)`; build a `pd.Series` indexed by column.
- Relabel row index to `gene_symbol` if available (fall back to probe).
- `sns.clustermap(mat_z, cmap='RdBu_r', center=0, col_colors=col_colors, figsize=(10, max(6, top*0.18)), xticklabels=True, yticklabels=True)` — clustermap runs the hierarchical clustering for both axes automatically.
- `g.savefig(output_path, dpi=200, bbox_inches='tight')`.

## `src/compare.py` — cross-dataset comparison

Inputs: two GEO accessions, group labels per dataset, output directory.

Steps:
1. Load and diff-express each dataset independently (reuse `diffex.py`'s pipeline). This gives `de_a`, `de_b`.
2. Join on `gene_symbol` (not probe ID — probes differ across platforms). Drop rows without a symbol on either side. If multiple probes map to the same symbol, keep the one with the smaller `adj_p_value` per dataset before joining.
3. Outputs in `result/compare_<a>_vs_<b>/`:
   - `shared_de_genes.csv` — genes significant (`adj_p_value < 0.05`) in both; columns `gene_symbol, log2FC_a, adj_p_a, log2FC_b, adj_p_b`.
   - `log2fc_scatter.png` — scatter of `log2FC_a` vs `log2FC_b` over the joined set, colored by "significant in both / one / neither". Annotate top genes by combined rank. Include Pearson r and Spearman ρ on the title.
   - `overlap_summary.txt` — counts: significant in A only, B only, both; Jaccard of the two significant sets; direction concordance (same sign among both-significant genes).

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
5. `python analyze_gene_expression/ge.py heatmap --accession GSE19804 --from-results result/GSE19804/de.csv --top 50`
   - Open `heatmap.png` — dendrogram should cluster tumor samples away from normal samples.
6. `python analyze_gene_expression/ge.py compare --accession-a GSE19804 --accession-b GSE10072 --group-a tumor --group-b normal`
   - `log2fc_scatter.png` should show positive Pearson r — the same biology across cohorts.
7. Re-run step 2 — should hit the GEOparse cache and finish in seconds (no re-download).

## Out of scope for v1

Called out so the scope is clear:
- RNA-seq count modeling (DESeq2 / edgeR / limma-voom) — v1 assumes microarray-style continuous expression already on log-scale or safely log2-transformable.
- Gene set enrichment (GSEA / ORA).
- Batch-effect correction (ComBat / limma::removeBatchEffect).
- Cross-platform probe-to-gene reconciliation beyond simple `gene_symbol` joining.
- Interactive dashboard — static PNGs only.
