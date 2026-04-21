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

Inputs: `expression`, `samples`, output directory.

Writes to `result/<accession>/eda/`:
- `summary.txt` — dataset shape, sample count, gene count, per-sample NaN fraction, value range, detected log-scale (yes/no via median < 30 heuristic).
- `sample_boxplots.png` — `seaborn.boxplot` of expression per sample (first 40 samples if more). Diagnoses batch effects.
- `sample_correlation_heatmap.png` — `sns.heatmap(expression.corr(), cmap='viridis')` with samples grouped by metadata `group` if present.
- `pca_scatter.png` — `sklearn.decomposition.PCA(n_components=2)` on `expression.T` (samples as rows). `matplotlib.pyplot.scatter` colored by `samples['group']`. Annotate % variance on axis labels.
- `gene_variance_hist.png` — histogram of per-gene variance; sanity check that highly variable genes exist.

No seaborn `clustermap` here — that lives in the heatmap module.

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
